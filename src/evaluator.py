from typing import Any, Dict, Tuple
import pandas as pd


class Evaluator:
    def __init__(
        self,
        row_retention_threshold: float = 0.95,
        distribution_threshold: float = 0.10,
    ):
        self.row_retention_threshold = row_retention_threshold
        self.distribution_threshold = distribution_threshold

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

        stats_passed, stats_reason = self._check_statistical_invariants(
            df_raw, df_clean
        )
        if not stats_passed:
            state.setdefault("execution_errors", []).append(f"Failed: {stats_reason}")
            return state

        semantic_passed, semantic_reason = self._check_semantic_fidelity(
            df_raw, df_clean, state
        )
        if not semantic_passed:
            state.setdefault("execution_errors", []).append(
                f"Failed: {semantic_reason}"
            )
            return state

        state["is_complete"] = True
        return state

    def _check_statistical_invariants(
        self, df_raw: pd.DataFrame, df_clean: pd.DataFrame
    ) -> Tuple[bool, str]:
        return

    def _build_semantic_prompt(
        self, raw_sample: pd.DataFrame, clean_sample: pd.DataFrame
    ) -> str:
        return f"""
        Review the data transformation.
        Did the transformation break the underlying semantic meaning of the data?
        
        RAW DATA SAMPLE:
        {raw_sample.to_dict(orient="records")}
        
        CLEANED DATA SAMPLE:
        {clean_sample.to_dict(orient="records")}
        
        Respond exactly with 'PASS' or 'FAIL: <exact semantic error reason>'.
        """

    def _check_semantic_fidelity(
        self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        prompt = self._build_semantic_prompt(df_raw.head(5), df_clean.head(5))
        response = llm_client.generate(prompt)
        if "FAIL" in response:
            return False, response.split("FAIL:")[-1].strip()

        True, ""
