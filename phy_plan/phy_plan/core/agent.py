"""
agent.py: LLM Agent wrapper

provide a unified LLM call interface, support OpenAI API and OpenRouter.
"""

import os
from typing import Optional
from openai import OpenAI


class LLMAgent:
    """
    lightweight LLM Agent
    
    support single-round and multi-round conversation, for task planning.
    support OpenAI API and OpenRouter.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        initialize Agent
        
        Args:
            model: model name (default gpt-4o-mini)
            api_key: API Key (default from environment variable)
            base_url: API base URL (default None, use OpenAI, OpenRouter use "https://openrouter.ai/api/v1")
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
                "environment variable, or pass api_key."
            )
        
        # priority: 1) base_url parameter 2) OPENAI_API_BASE environment variable 3) other logic
        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENAI_API_BASE"):
            self.base_url = os.getenv("OPENAI_API_BASE")
        elif os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            # if only OPENROUTER_API_KEY, default use OpenRouter
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            # default use OpenAI (None)
            self.base_url = None
        
        # create client
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # multi-round conversation history
        self.messages = []
    
    @property
    def name(self) -> str:
        return f"LLMAgent({self.model})"
    
    def llm_call(
        self,
        system_content: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> str:
        """
        single-round LLM call
        
        Args:
            system_content: System message (define role and constraints)
            user_prompt: User message (specific task)
            temperature: temperature parameter (default 0.0 to ensure deterministic output)
            max_tokens: maximum output token number
            
        Returns:
            LLM response text
            
        Raises:
            RuntimeError: if response is truncated due to length limit
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # check if response is truncated due to length limit
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"[WARNING] LLM response was truncated due to max_tokens limit ({max_tokens})")
            print(f"[WARNING] Consider increasing max_tokens or simplifying the prompt")
            # still return response, but record warning
        
        return response.choices[0].message.content
    
    def init_conversation(self, system_content: str) -> None:
        """
        initialize conversation
        
        clear history and set system prompt.
        
        Args:
            system_content: System message (define role and constraints)
        """
        self.messages = [
            {"role": "system", "content": system_content}
        ]
    
    def chat(
        self,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> str:
        """
        multi-round conversation (based on history)
        
        Args:
            user_prompt: User message
            temperature: temperature parameter
            max_tokens: maximum output token number
            
        Returns:
            LLM response text
            
        Raises:
            RuntimeError: if response is truncated due to length limit
        """
        # add user message
        self.messages.append({"role": "user", "content": user_prompt})
        
        # call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # check if response is truncated due to length limit
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            print(f"[WARNING] LLM response was truncated due to max_tokens limit ({max_tokens})")
            print(f"[WARNING] Consider increasing max_tokens or simplifying the prompt")
            print(f"[WARNING] Partial response may cause JSON parsing errors")
            # still return response, but record warning
        
        # get response
        assistant_message = response.choices[0].message.content
        
        # add assistant message to history
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def reset(self) -> None:
        """clear conversation history"""
        self.messages = []
    
    def __repr__(self) -> str:
        base_info = f", base_url={self.base_url}" if self.base_url else ""
        return f"LLMAgent(model={self.model}{base_info})"
