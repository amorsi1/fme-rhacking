# Adapted from Countdown-Code by Zohaib Khan et al.
# https://github.com/zohaib-khan5040/Countdown-Code @ 170ee81
# Upstream ships no LICENSE file; used here for internal research under the
# original authors' copyright. See CLAUDE.md for local modifications.

from typing import Dict, List
from openai import AsyncOpenAI

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key
        )
    
    async def get_response(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.responses.create(
            model=model,
            input=messages,
            **kwargs
        )
        return response