from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor


class Evaluator:
    def __init__(
        self,
        client,
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

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

        global_checks = [
            self._check_schema(df_raw, df_clean),
            self._check_missing_values(df_raw, df_clean, state),
            self._check_statistical_invariants(df_raw, df_clean),
            self._check_domain_constraints(df_clean, state),
        ]

        evaluation_errors = {
            "global_checks": {
                "schema": self._check_schema(df_raw, df_clean, state),
                "row_retention": self.check_row_retention(df_raw, df_clean),
            },
            "missing_values": self._check_missing_values(df_clean),
            "domain_constraints": self._check_domain_constraints(
                df_raw, df_clean, state["metadata"]
            ),
            "statistical_invariants": self._check_statistical_invariants(
                df_raw, df_clean, state["metadata"]
            ),
            "categorical_distribution": self._check_categorical_distribution(
                df_raw, df_clean
            ),
        }

        state["evaluation"] = {
            "passed": state["is_complete"] == len(state["execution_errors"]) == 0,
        }

        state["is_complete"] = len(state["execution_errors"]) == 0
        return state

    def _check_schema(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        transformations: Dict[str, Any],
    ) -> Tuple[bool, str]:
        is_complete = False
        reason = ""

        # TODO:
        # 1. Determine expected columns after transformations.
        # 2. Check unexpected columns.
        # 3. Check unexpectedly removed columns.
        # 4. Validate expected dtype changes.

        new_columns = [
            new_col
            for _, transformation in transformations.items()
            for new_col in transformation["expected columns"]
        ]

        columns_not_in_df = list(set(new_columns).difference(df_clean.columns))
        unexpected_columns = list(set(df_clean.columns).difference(new_columns))
        dtype_mismatch = []

        for col in df_raw.columns:
            if "Casting" in transformations[col]["types"]:
                if transformations[col]["dtype"] != df_clean[col].dtype:
                    dtype_mismatch.append[col]

        transformations
        return (
            (
                not len(columns_not_in_df)
                and not len(columns_not_in_df)
                and not len(dtype_mismatch)
            ),
            f"missing columns from dataframe: {columns_not_in_df}, unexpected columsn in dataframe: {unexpected_columns}, columns with dtype mismatch:{dtype_mismatch} ",
        )

    def _evaluate_column(
        self,
        column: str,
        raw: pd.Series,
        clean: pd.Series,
        metadata: Dict[str, Any],
        state: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Run checks appropriate for an individual column."""
        failures = []
        # Missing values
        passed, reason = self._check_missing_values(
            raw,
            clean,
            metadata,
        )
        if not passed:
            failures.append(
                {
                    "scope": "column",
                    "column": column,
                    "check": "missing_values",
                    "reason": reason,
                }
            )

        # Statistical invariants
        passed, reason = self._check_statistical_invariants(
            raw,
            clean,
            metadata,
        )
        if not passed:
            failures.append(
                {
                    "scope": "column",
                    "column": column,
                    "check": "statistical_invariants",
                    "reason": reason,
                }
            )

        # Domain constraints
        passed, reason = self._check_domain_constraints(
            clean,
            metadata,
        )
        if not passed:
            failures.append(
                {
                    "scope": "column",
                    "column": column,
                    "check": "domain_constraints",
                    "reason": reason,
                }
            )
            return failures

    def _check_missing_values(self, df_clean: pd.DataFrame) -> Tuple[bool, str]:

        if not df_clean.notna().all(axis=None):
            return False, f"{df_clean.isna().sum().to_dict()}"

        return True, ""

    def _check_domain_constraints(
        self, df_clean: pd.DataFrame, metadata
    ) -> Tuple[bool, str]:
        is_complete = False
        reason = ""
        return True, ""

    def _check_statistical_invariants(
        self, df_raw: pd.DataFrame, df_clean: pd.DataFrame
    ) -> Tuple[bool, str]:
        is_complete = False
        reason = ""
        return True, ""

    def _check_categorical_distribution(
        self, raw: pd.Series, clean: pd.Series
    ) -> Tuple[bool, dict]:

        raw_dist = raw.value_counts(normalize=True)
        clean_dist = clean.value_counts(normalize=True)

        categories = raw_dist.index.union(clean_dist.index)

        raw_dist = raw_dist.reindex(categories, fill_value=0)
        clean_dist = clean_dist.reindex(categories, fill_value=0)

        difference = (raw_dist - clean_dist).abs().max()

        if difference > self.distribution_threshold:
            return (False, f"Maximum category distribution shift: {difference:.2%}")

        return True, {}
