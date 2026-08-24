import logging
import os
import time
import docker

# import uuid
from typing import Optional
from typing import Dict, Any
import json
from openai import OpenAI

import pandas as pd
from evaluator import Evaluator
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cosine_similarity
import numpy as np
import torch

TYPE_DESCRIPTIONS = {
    "quantitative": "A numeric measurement where arithmetic operations are meaningful.",
    "categorical": "A variable containing discrete unordered categories.",
    "ordinal": "A variable containing categories with meaningful ordering.",
    "datetime": "A date or timestamp representing time.",
    "date_component": "A standalone part of a date such as just days, just months, or just years.",
    "identifier": "A value used primarily to uniquely identify an entity.",
    "text": "Free-form natural language or descriptive text.",
}

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when session ID resolution fails"""

    pass


class Orchestrator:
    """
    Initialize orchestrator

    Args:
        api_key: OpenAI API key
        model: LLM model to use
        system_prompt: Optional system prompt for orchestration
        log_level: Logging level for LLM input/output logging. Can be:
                  - String: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                  - Integer: logging.DEBUG (10), logging.INFO (20), etc.
                  - Default: logging.INFO (20)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        log_level: int | str = logging.INFO,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.api_key = api_key
        self.model = model
        self.system_prompt = (
            system_prompt
            if system_prompt
            else """
                You are an autonomous data engineering agent. Your objective is to write a deterministic Python script to clean a dataset.

                ### EXECUTION ENVIRONMENT & CONSTRAINTS
                1. Sandbox: The code will execute in an isolated container. 
                2. Dynamic Paths: DO NOT hardcode file paths. Read the input file path from `sys.argv[1]` and write the final output to `sys.argv[2]`.
                3. Libraries: Use standard libraries, `pandas`, or `duckdb`.
                4. Scope: Fix only the requested anomalies. Do not mutate valid columns.
                5. Logging: Print only brief summary statistics to stdout.

                ### OUTPUT FORMAT
                Output strictly valid, executable Python code. 
                Do not output conversational text, explanations, or markdown code blocks. Start immediately with:

                import sys
                import pandas as pd

                if __name__ == "__main__":
                    input_path = sys.argv[1]
                    output_path = sys.argv[2]

                """
        )
        self.client = OpenAI(api_key=self.api_key)

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        error_context = ""
        if state.get("execution_errors"):
            error_context = f"""
            ### PREVIOUS EXECUTION ERRORS
            The previous script failed. Analyze the traceback below and correct the logic:
            {state["execution_errors"][-1]}
            """

        user_prompt = f"""
        ### FIRST 10 ROWS:
        {state["metadata"]["preview"]}
        
        ### Data Profile:
        {state["metadata"]["columns_metadata"]}

        {error_context}
        
        Write the Python script to resolve these anomalies.
        """
        return user_prompt.strip()

    def _execute_in_docker_sandbox(
        self, generated_code: str, data_file_path: str, output_file_path: str
    ) -> dict:
        """
        Executes LLM-generated code in an isolated Docker container.
        """
        client = docker.from_env()

        script_path = os.path.abspath("./temp_agent_script.py")
        with open(script_path, "w") as f:
            f.write(generated_code)

        input_path = os.path.abspath(data_file_path)
        input_filename = os.path.basename(input_path)

        output_dir = os.path.dirname(os.path.abspath(output_file_path))
        output_filename = os.path.basename(output_file_path)

        try:
            container = client.containers.run(
                image="python:3.11-slim",  # Use a minimal image
                command=[
                    "python",
                    "/app/script.py",
                    f"/data/{input_filename}",
                    f"/output/{output_filename}",
                ],
                volumes={
                    script_path: {"bind": "/app/script.py", "mode": "ro"},
                    input_path: {"bind": f"/{input_filename}", "mode": "rw"},
                    output_dir: {"bind": "/output", "mode": "rw"},
                },
                working_dir="/app",
                network_disabled=True,
                mem_limit="2g",
                cpu_period=100000,
                cpu_quota=50000,
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
            )

            return {"status": "success", "output": container.decode("utf-8")}

        except docker.errors.ContainerError as e:
            return {"status": "error", "error_log": e.stderr.decode("utf-8")}
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    # @staticmethod
    def describe_column(
        self,
        df,
        column,
        class_approx,
        min_unique_values=20,
        unique_ratio_threshold=0.05,
    ):
        col = df[column]
        data_type = col.dtype
        cardinality_ratio = col.nunique() / len(df) if len(df) > 0 else 0

        col_info = {
            "datatype": str(data_type),
            "type": class_approx,
            "missing_count": int(col.isna().sum()),
            "unique_count": len(col.nunique()),
            "cardinality_ratio": round(cardinality_ratio, 4),
        }
        description = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

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
        elif class_approx == "categorical":
            col_info.update(
                {
                    "description": "Customer rating of the product on a 1–5 scale.",
                    "categories": col.value_counts().to_dict(),
                    "sample_values": column.dropna()
                    .sample(min(12, column.dropna().shape[0]), random_state=42)
                    .tolist(),
                }
            )
            pass

        elif class_approx == "ordinal":

            col_info.update(
                {
                    "description": "Customer rating of the product on a 1–5 scale.",
                    "categories": col.value_counts().to_dict(),
                    "sample_values": column.dropna()
                    .sample(min(12, column.dropna().shape[0]), random_state=42)
                    .tolist(),
                }
            )

            pass

        elif class_approx == "identifier":
            pass

        elif class_approx == "text":
            pass  # call llm to get description

        col_info["distinct_values"] = [str(val) for val in col.unique().tolist()]

        return {}

    def _extract_initial_metadata(self, file):
        df = pd.read_csv(file, parse_dates=True)
        output_lines = {}

        output_lines["preview"] = df.head(10).to_dict(orient="records")
        output_lines["columns_metadata"] = {}

        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

        class_inputs = [
            f"search_query: {label}, which is '{value}'"
            for label, value in TYPE_DESCRIPTIONS.keys()
        ]
        embeddings = model.encode(
            class_inputs, convert_to_tensor=True, normalize_embeddings=True
        )

        for column in df.columns:
            samples = (
                column.dropna()
                .sample(min(12, column.dropna().shape[0]), random_state=42)
                .tolist()
            )
            column_embeddings = model.encode(
                f""" Column name: {column.name}
                Sample values: {samples}
                """,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            similarity_scores = cosine_similarity(embeddings, column_embeddings)
            best_class_idx = torch.argmax(similarity_scores).item()
            class_approx = TYPE_DESCRIPTIONS[best_class_idx]
            col_info = Orchestrator.describe_column(df, column, class_approx)
            output_lines["columns_metadata"][column] = col_info

        return {
            "metadata": {
                "preview": json.dumps(output_lines["preview"], indent=2),
                "columns_metadata": json.dumps(
                    output_lines["columns_metadata"], indent=2
                ),
            }
        }

    def preprocess(self, file_path, output_file_path, max_step=100):
        eval = Evaluator(self.client)
        state = {"metadata": self._extract_initial_metadata(file_path)}
        state["execution_errors"] = []

        for step in range(max_step):
            print(f"--- Iteration {step + 1} ---")
            prompt = self._build_prompt(state)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            print(response.choices[0].message.content)

            raw_code = response.choices[0].message.content
            code = raw_code.replace("```python", "").replace("```", "").strip()
            state["generated_code"] = code

            status_output = self._execute_in_docker_sandbox(
                code, file_path, output_file_path
            )
            status = status_output["status"]

            if status == "error":
                error_msg = f"Runtime Error: {status}"
                state["execution_errors"].append(error_msg)
                logging.info(error_msg)
                time.sleep(19)
                continue

            output = status_output["output"]
            state["is_complete"] = eval.evaluate(state)

            logging.info(output)
            if state.get("is_complete"):
                logging.info("Pipeline successfully validated.")
                break
            else:
                logging.info(f"Validation Failed: {state['execution_errors'][-1]}")
                time.sleep(19)

        return
