import json
import logging
from typing import Dict, Optional

from openai import OpenAI
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cosine_similarity
import torch
from transformers import Any

from prompts import METADATA_EXTRACTOR_PROMPT

TYPE_DESCRIPTIONS = {
    "quantitative": "A numeric measurement where arithmetic operations are meaningful.",
    "categorical": "A variable containing discrete unordered categories.",
    "ordinal": "A variable containing categories with meaningful ordering.",
    "datetime": "A date or timestamp representing time.",
    "date_component": "A standalone part of a date such as just days, just months, or just years.",
    "identifier": "A value used primarily to uniquely identify an entity.",
    "text": "Free-form natural language or descriptive text.",
}


class MetadataExtractor:

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        log_level: int | str = logging.INFO,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.client = client
        self.model = model
        self.system_prompt = (
            system_prompt if system_prompt else METADATA_EXTRACTOR_PROMPT
        )

        self.sentence_transformer_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

        class_inputs = [
            f"search_query: {label}, which is '{value}'"
            for label, value in TYPE_DESCRIPTIONS.keys()
        ]
        self.embeddings = self.sentence_transformer_model.encode(
            class_inputs, convert_to_tensor=True, normalize_embeddings=True
        )

    def _build_prompt(self, col_info: Dict[str, Any], col_sample_values) -> str:

        user_prompt = f"""
            Below is the extracted profile of the dataset. 
            
            ### Data Profile:
            {col_info}
            sample values include:
            {col_sample_values}

            Use this information to perform the requested metadata extraction.
            """
        return user_prompt.strip()

    def describe_column(
        self,
        df,
        column,
        class_approx,
        min_unique_values=20,
    ):
        col = df[column]
        data_type = col.dtype
        cardinality_ratio = col.nunique() / len(df) if len(df) > 0 else 0
        sample_values = (
            col.dropna()
            .sample(min(min_unique_values, len(col.dropna())), random_state=42)
            .tolist()
        )

        col_info = {
            "datatype": str(data_type),
            "type": class_approx,
            "missing_count": int(col.isna().sum()),
            "unique_count": len(col.nunique()),
            "cardinality_ratio": round(cardinality_ratio, 4),
            "sample_values": sample_values,
        }

        if class_approx == "datetime":
            col_info["date_range"] = {
                "min": str(col.min()),
                "max": str(col.max()),
            }
        elif class_approx == "quantitative":
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            col_info.update(
                {
                    "min": col.min(),
                    "q1": q1,
                    "median": col.median(),
                    "mean": col.mean(),
                    "q3": q3,
                    "max": col.max(),
                    "iqr": q3 - q1,
                }
            )
        elif class_approx in ["categorical", "ordinal"]:
            col_info.update(
                {
                    "categories": col.value_counts().to_dict(),
                }
            )
        elif class_approx in ["identifier", "text"]:
            col_info.update(
                {
                    "sample_values": column.dropna()
                    .sample(min(12, column.dropna().shape[0]), random_state=42)
                    .tolist(),
                }
            )

        prompt = self._build_prompt(col_info, sample_values)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        col_info["llm_description"] = json.loads(response.choices[0].message.content)

        return col_info

    def extract(self, file):
        df = pd.read_csv(file, parse_dates=True)
        output_lines = {}

        output_lines["preview"] = df.head(10).to_dict(orient="records")
        output_lines["columns_metadata"] = {}

        for column in df.columns:
            samples = (
                column.dropna()
                .sample(min(12, column.dropna().shape[0]), random_state=42)
                .tolist()
            )
            column_embeddings = self.sentence_transformer_model.encode(
                f""" Column name: {column.name}
                Sample values: {samples}
                """,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            similarity_scores = cosine_similarity(self.embeddings, column_embeddings)
            best_class_idx = torch.argmax(similarity_scores).item()
            class_approx = TYPE_DESCRIPTIONS[best_class_idx]
            col_info = self.describe_column(df, column, class_approx)
            output_lines["columns_metadata"][column] = col_info

        return {
            "metadata": {
                "preview": json.dumps(output_lines["preview"], indent=2),
                "columns_metadata": json.dumps(
                    output_lines["columns_metadata"], indent=2
                ),
            }
        }

    pass
