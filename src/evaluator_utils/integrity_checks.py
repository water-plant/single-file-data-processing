from __future__ import annotations

from typing import Any, Dict, List, Tuple
from evaluator_utils.result import CheckResult
import pandas as pd


class IntegrityChecks:
    """collection of pure utility functions for preprocessing steps"""

    DEFAULT_ROW_RETENTION_THRESHOLD = 0.99

    @staticmethod
    def check_schema(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        transformations: List[Dict[str, Any]],
    ) -> CheckResult:

        changed_columns = IntegrityChecks._changed_columns(transformations)
        all_output_columns = set(df_raw.columns) - set(changed_columns)

        for operation in transformations:
            all_output_columns.update(operation.get("output_columns", []))

        actual_columns = set(df_clean.columns)

        columns_not_in_df = sorted(all_output_columns - actual_columns)
        unexpected_columns = sorted(actual_columns - all_output_columns)

        dtype_mismatch = []
        for operation in transformations:
            col = operation["column"]
            if "casting" == operation["type"]:
                expected_dtype = pd.api.types.pandas_dtype(operation["dtype"])
                if expected_dtype["dtype"] != df_clean[col].dtype:
                    dtype_mismatch.append[col]
                    dtype_mismatch.append(
                        {
                            "column": col,
                            "expected": str(expected_dtype),
                            "actual": str(df_clean[col].dtype),
                        }
                    )

        if not columns_not_in_df and not unexpected_columns and not dtype_mismatch:
            return CheckResult.ok("schema")

        return CheckResult.fail(
            "schema",
            f"missing columns from dataframe: {columns_not_in_df}, unexpected columsn in dataframe: {unexpected_columns}, columns with dtype mismatch:{dtype_mismatch}",
            details={
                "missing_columns": columns_not_in_df,
                "unexpected_columns": unexpected_columns,
                "dtype_mismatch": dtype_mismatch,
            },
        )

    @staticmethod
    def check_missing_values(df_clean: pd.DataFrame) -> Tuple[bool, str]:
        null_counts = df_clean.isna().sum()
        null_counts = null_counts[null_counts > 0].to_dict()

        if null_counts:
            return CheckResult.fail(
                "missing_values",
                f"columns contain null values: {null_counts}",
                details={"null_counts": null_counts},
            )

        return CheckResult.ok("missing_values")

    @staticmethod
    def row_retention(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        threshold=DEFAULT_ROW_RETENTION_THRESHOLD,
    ):
        if len(df_raw) == 0:
            return CheckResult.ok(
                "row_retention",
                details={"raw_rows": 0, "clean_rows": len(df_clean)},
            )

        retention = len(df_clean) / len(df_raw)
        if retention < threshold:
            return CheckResult.fail(
                "row_retention",
                f"row retention {retention:.3%} is below threshold {threshold:.3%}",
                details={
                    "raw_rows": len(df_raw),
                    "clean_rows": len(df_clean),
                    "retention": retention,
                    "threshold": threshold,
                },
            )

        return CheckResult.ok(
            "row_retention",
            details={
                "raw_rows": len(df_raw),
                "clean_rows": len(df_clean),
                "retention": retention,
                "threshold": threshold,
            },
        )

        pass

    @staticmethod
    def distribution_preservation(state):
        pass

    @staticmethod
    def ensure_unchanged_columns(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        operations: List[Dict[str, Any]],
    ) -> CheckResult:
        changed_columns = IntegrityChecks._changed_columns(operations)
        unchanged_columns = list(df_raw.columns.difference(changed_columns))
        unexpectedly_changed_columns = []
        for col in unchanged_columns:
            if not df_raw[col].equals(df_clean[col]):
                unexpectedly_changed_columns.append(col)

        if not unexpectedly_changed_columns:
            return CheckResult.fail(
                "ensure_unchanged_columns",
                f"unexpectedly changed in output dataframe: {unexpectedly_changed_columns}",
                details={
                    "unexpectedly_changed_columns": unexpectedly_changed_columns,
                },
            )

        return CheckResult.ok("ensure_unchanged_columns")

    @staticmethod
    def check_unexpected_row_duplicates(
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
    ) -> CheckResult:
        raw_duplicates = df_raw.duplicated().sum()
        clean_duplicates = df_clean.duplicated().sum()
        if clean_duplicates > raw_duplicates:
            return CheckResult(
                passed=False,
                message="Preprocessing introduced additional duplicate rows.",
            )

        return CheckResult(
            passed=True, message="No unexpected increase in duplicate rows."
        )

    @staticmethod
    def _changed_columns(operations: Any | None) -> set[str]:
        changed_columns = set()
        for operation in operations:
            changed_columns.add(operation["column"])

        return changed_columns
