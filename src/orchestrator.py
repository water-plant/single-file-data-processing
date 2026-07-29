import logging
import os
import docker
import uuid
from typing import Optional, Callable, List
import subprocess
from typing import Dict, Any
import json

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
        self.system_prompt = system_prompt

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        # Format error context only if errors exist
        error_context = ""
        if state.get("execution_errors"):
            error_context = f"""
            ### PREVIOUS EXECUTION ERRORS
            The previous script failed. Analyze the traceback below and correct the logic:
            {state['execution_errors'][-1]}
            """

        prompt = f"""
            You are an expert data developer tasked with iteratively clean a dataset.
            Your objective is to write a deterministic Python script to clean a dataset based on its profile and identified anomalies.

            ### SYSTEM CONTEXT
            - Schema & Data Profile: {json.dumps(state.get('schema_info', {}), indent=2)}
            - Anomalies to Resolve: {json.dumps(state.get('detected_anomalies', []), indent=2)}
            {error_context}

            ### EXECUTION ENVIRONMENT & CONSTRAINTS
            1. Sandbox: The code will execute in an isolated container. 
            2. Dynamic Paths: DO NOT hardcode file paths. Read the input file path from `sys.argv[1]` and write the final output to `sys.argv[2]`.
            3. Libraries: Use standard libraries, `pandas`, or `duckdb`.
            4. Scope: Fix only the anomalies listed. Do not mutate valid columns.
            5. Logging: Print only brief summary statistics (e.g., row counts before/after) to stdout.

            ### OUTPUT FORMAT
            Output strictly valid, executable Python code. 
            Do not output conversational text, explanations, or markdown code blocks (e.g., ```python).

            import sys
            import pandas as pd

            if __name__ == "__main__":
                input_path = sys.argv[1]
                output_path = sys.argv[2]
                
                # Write your transformation logic below
        """
        return prompt.strip()

    def execute_in_docker_sandbox(
        self, generated_code: str, data_file_path: str
    ) -> dict:
        """
        Executes LLM-generated code in an isolated Docker container.
        """
        client = docker.from_env()

        script_path = os.path.abspath("./temp_agent_script.py")
        with open(script_path, "w") as f:
            f.write(generated_code)

        absolute_data_path = os.path.abspath(data_file_path)

        try:
            container = client.containers.run(
                image="python:3.11-slim",  # Use a minimal image
                command=["python", "/app/script.py"],
                volumes={
                    script_path: {"bind": "/app/script.py", "mode": "ro"},
                    absolute_data_path: {"bind": self.input, "mode": "ro"},
                    os.path.abspath("./output"): {"bind": "/output", "mode": "rw"},
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

    def _extract_initial_metadata(self, file):
        table_names = self.data_connection.list_tables()
        if not table_names:
            return {}
        return {name: self.data_connection.table(name).schema() for name in table_names}

    def preprocess(self, file):
        return
