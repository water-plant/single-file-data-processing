from typing import Any, Dict, List, Tuple
from evaluator_utils.integrity_checks import IntegrityChecks
from evaluator_utils.result import CheckResults
import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor


class Evaluator:
    def __init__(
        self,
        row_retention_threshold: float = 0.95,
        distribution_threshold: float = 0.10,
    ):
        self.row_retention_threshold = row_retention_threshold
        self.distribution_threshold = distribution_threshold

        self._transformation_checks = {
            "one_hot_encoding": self._check_one_hot_encoding,
            "ordinal_encoding": self._check_ordinal_encoding,
            "standardization": self._check_standardization,
            "min_max_scaling": self._check_min_max_scaling,
            "imputation": self._check_imputation,
        }

    def evaluate(
        self, state: Dict[str, Any], operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Runs evaluation on the agent's output."""

        if state.get("execution_errors"):
            return state

        raw_path = state.get("dataset_path")
        clean_path = state.get("clean_dataset_path")

        try:
            df_raw = pd.read_csv(raw_path)
            df_clean = pd.read_csv(clean_path)
        except Exception as e:
            state.setdefault("execution_errors", []).append(
                f"Failed: Output missing or corrupt. {str(e)}"
            )
            return state

        results = CheckResults()

        results.add(IntegrityChecks._check_schema(df_raw, df_clean, operations))
        results.add(IntegrityChecks.row_retention(df_raw, df_clean))
        results.add(IntegrityChecks.null_values(df_clean))
        results.add(
            IntegrityChecks.distribution_preservation(
                df_raw,
                df_clean,
                operations,
            )
        )
        results.add(
            IntegrityChecks.ensure_unchanged_columns(
                df_raw,
                df_clean,
                operations,
            )
        )

        state["evaluation"] = results.to_dict()
        state["is_complete"] = results.passed
        state["score"] = results.score

        return state

    # def _check_schema(
    #     self,
    #     df_raw: pd.DataFrame,
    #     df_clean: pd.DataFrame,
    #     transformations: Dict[str, Any],
    # ) -> Tuple[bool, str]:
    #     # TODO:
    #     # 1. Determine expected columns after transformations.
    #     # 2. Check unexpected columns.
    #     # 3. Check unexpectedly removed columns.
    #     # 4. Validate expected dtype changes.

    #     all_output_columns = [
    #         new_col
    #         for operation in transformations
    #         for new_col in operation["output_columns"]
    #     ]

    #     columns_not_in_df = list(set(all_output_columns).difference(df_clean.columns))
    #     unexpected_columns = list(set(df_clean.columns).difference(all_output_columns))
    #     dtype_mismatch = []

    #     for col in df_raw.columns:
    #         if "Casting" in transformations[col]["types"]:
    #             if transformations[col]["dtype"] != df_clean[col].dtype:
    #                 dtype_mismatch.append[col]

    #     return (
    #         (
    #             not len(columns_not_in_df)
    #             and not len(columns_not_in_df)
    #             and not len(dtype_mismatch)
    #         ),
    #         f"missing columns from dataframe: {columns_not_in_df}, unexpected columsn in dataframe: {unexpected_columns}, columns with dtype mismatch:{dtype_mismatch} ",
    #     )

    # def _check_missing_values(self, df_clean: pd.DataFrame) -> Tuple[bool, str]:

    #     if not df_clean.notna().all(axis=None):
    #         return False, f"{df_clean.isna().sum().to_dict()}"

    #     return True, ""

    # def _check_domain_constraints(
    #     self, df_clean: pd.DataFrame, metadata
    # ) -> Tuple[bool, str]:
    #     is_complete = False
    #     reason = ""

    #     for col in df_clean.columns:
    #         pass

    #     return True, ""

    # def _check_categorical_distribution(
    #     self, raw: pd.Series, clean: pd.Series
    # ) -> Tuple[bool, dict]:

    #     raw_dist = raw.value_counts(normalize=True)
    #     clean_dist = clean.value_counts(normalize=True)

    #     categories = raw_dist.index.union(clean_dist.index)

    #     raw_dist = raw_dist.reindex(categories, fill_value=0)
    #     clean_dist = clean_dist.reindex(categories, fill_value=0)

    #     difference = (raw_dist - clean_dist).abs().max()

    #     if difference > self.distribution_threshold:
    #         return (False, f"Maximum category distribution shift: {difference:.2%}")

    #     return True, {}
