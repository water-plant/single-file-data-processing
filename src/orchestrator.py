from __future__ import annotations

import logging
import os
import time
import docker

# import uuid
from typing import List, Optional
from typing import Dict, Any
import json
from openai import OpenAI

from planner import PlanGenerator
from evaluator import Evaluator
from metadata_extraction import MetadataExtractor

import numpy as np

from prompts import CODE_GENERATOR_PROMPT

logger = logging.getLogger(__name__)


# class SessionNotFoundError(Exception):
#     """Raised when session ID resolution fails"""

#     pass


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
        self.code_system_prompt = (
            system_prompt if system_prompt else CODE_GENERATOR_PROMPT
        )
        self.client = OpenAI(api_key=self.api_key)

    def _build_prompt(
        self, metadata, execution_errors, operations: List[Dict[str, Any]]
    ) -> str:
        error_context = ""
        if execution_errors:
            error_context = f"""
            ### PREVIOUS EXECUTION ERRORS
            The previous script failed. Analyze the traceback below and correct the logic:
            {execution_errors[-1]}
            """

        user_prompt = f"""
        ### Ordered List Of Operations:
        {operations}
        
        ### FIRST 10 ROWS:
        {metadata["preview"]}
        
        ### Data Profile:
        {metadata["columns_metadata"]}

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

    def preprocess(
        self,
        file_path,
        output_file_path,
        user_request: str | None = None,
        max_step: int = 100,
    ):
        eval = Evaluator(self.client)
        metadata_extractor = MetadataExtractor(self.client)
        planner = PlanGenerator(self.client)

        original_metadata = metadata_extractor.extract(file_path)
        execution_errors = ""

        for step in range(max_step):
            print(f"--- Iteration {step + 1} ---")
            preprocessing_plan = planner.generate_plan(user_request, original_metadata)
            # state.update(preprocessing_plan)
            prompt = self._build_prompt(
                original_metadata,
                execution_errors,
                preprocessing_plan,
            )
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
            # state["generated_code"] = code

            status_output = self._execute_in_docker_sandbox(
                code, file_path, output_file_path
            )
            status = status_output["status"]

            if status == "error":
                execution_errors = f"Runtime Error: {status}"
                logging.info(execution_errors)
                time.sleep(19)
                continue

            output = status_output["output"]
            is_complete = eval.evaluate()

            logging.info(output)

            if is_complete:
                logging.info("Pipeline successfully validated.")
                break
            else:
                logging.info(f"Validation Failed: {execution_errors}")
                time.sleep(19)

        return
