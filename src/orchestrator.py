import logging
import os
import docker
import uuid
from typing import Optional, Callable, List
import subprocess
from typing import Dict, Any
import json
from openai import OpenAI

import openai
from .evaluator import Evaluator

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
            if not system_prompt
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

        self.client = OpenAI(api_key=api_key)

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        error_context = ""
        if state.get("execution_errors"):
            error_context = f"""
            ### PREVIOUS EXECUTION ERRORS
            The previous script failed. Analyze the traceback below and correct the logic:
            {state['execution_errors'][-1]}
            """

        user_prompt = f"""
        ### SYSTEM CONTEXT
        - Schema & Data Profile: {json.dumps(state.get('schema_info', {}), indent=2)}
        - Anomalies to Resolve: {json.dumps(state.get('detected_anomalies', []), indent=2)}
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

    def _extract_initial_metadata(self, file):
        table_names = self.data_connection.list_tables()
        if not table_names:
            return {}
        return {name: self.data_connection.table(name).schema() for name in table_names}

    def preprocess(self, file_path, max_step=100):
        eval = Evaluator(self.api_key)
        state = {}
        for step in range(max_step):
            print(f"--- Iteration {step + 1} ---")
            state["metadata"] = self._extract_initial_metadata(file_path)
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
            status_output = self._execute_in_docker_sandbox(code, file_path)
            status = status_output["status"]
            output = (
                status_output["output"]
                if status == "status"
                else status_output["error_log"]
            )
            logging.info(output)
            if status == "success":
                eval.evaluate(output)

        return
