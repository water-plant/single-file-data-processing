from typing import Any, Dict, Tuple
import pandas as pd
import openai


class Evaluator:
    def __init__(
        self,
        client,
        row_retention_threshold: float = 0.95,
        distribution_threshold: float = 0.10,
    ):
        self.row_retention_threshold = row_retention_threshold
        self.distribution_threshold = distribution_threshold
        self.client = client

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

        checks = [
            self._check_schema(df_raw, df_clean),
            self._check_row_changes(df_raw, df_clean, state),
            self._check_missing_values(df_raw, df_clean, state),
            self._check_statistical_invariants(df_raw, df_clean),
            self._check_domain_constraints(df_clean, state),
        ]
        for i in range(len(checks)):
            if not checks[i][0]:
                state.setdefault("execution_errors", []).append(
                    f"Failed: {checks[0][1]}"
                )
            # return state

        # semantic_passed, semantic_reason = self._check_semantic_fidelity(
        #     df_raw, df_clean, state
        # )
        # if not semantic_passed:
        #     state.setdefault("execution_errors", []).append(
        #         f"Failed: {semantic_reason}"
        #     )
        #     return state

        state["is_complete"] = len(state["execution_errors"]) == 0
        return state

    def _check_statistical_invariants(
        self, df_raw: pd.DataFrame, df_clean: pd.DataFrame
    ) -> Tuple[bool, str]:
        is_complete = False
        reason = ""
        return (not is_complete, reason)
