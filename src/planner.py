import json
import logging
from typing import Any, Dict, Optional

import openai
from prompts import PLANNER_PROMPT


class PlanGenerator:

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        log_level: int | str = logging.INFO,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.client = client
        self.model = model

        self.system_prompt = system_prompt if system_prompt else PLANNER_PROMPT

    def _build_prompt(self, user_request: str, metadata: Dict[str, Any]) -> str:

        user_prompt = f"""
                Generate a preprocessing plan for the following request. 
                
                ## User Request 
                {user_request} 

                ### FIRST 10 ROWS:
                {metadata["preview"]}
                
                ## Dataset Metadata 
                {metadata} 
                
                Return a valid JSON object following the required output schema. 
                Use the dataset metadata to identify relevant columns and determine applicable parameters. 
                
                Do not assume missing requirements or introduce unrequested transformations. 
                If essential information is ambiguous or missing, set "requires_clarification" to true and provide clarification questions instead of generating a plan.
                
                """
        return user_prompt.strip()

    def generate_plan(self, user_request: str, state: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self._build_prompt(user_request, state),
                },
            ],
        )

        return json.loads(response.choices[0].message.content)
