import logging
import os
import docker
import uuid
from typing import Optional, Callable, List
import subprocess

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

    def load_file(self, file):
        self.input = file
        return

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

    def _extract_initial_metadata(self):
        table_names = self.data_connection.list_tables()
        if not table_names:
            return {}
        return {name: self.data_connection.table(name).schema() for name in table_names}
