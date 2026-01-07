"""
agent.py: LLM Agent 封装

提供统一的 LLM 调用接口，支持 OpenAI API 和 OpenRouter。
"""

import os
from typing import Optional
from openai import OpenAI


class LLMAgent:
    """
    轻量级 LLM Agent
    
    支持单轮对话，用于任务规划。
    支持 OpenAI API 和 OpenRouter。
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化 Agent
        
        Args:
            model: 模型名称（默认 gpt-4o-mini）
            api_key: API Key（默认从环境变量读取）
            base_url: API base URL（默认 None 使用 OpenAI，OpenRouter 使用 "https://openrouter.ai/api/v1"）
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
                "environment variable, or pass api_key."
            )
        
        # 优先级：1) 传入的 base_url 参数 2) OPENAI_API_BASE 环境变量 3) 其他逻辑
        if base_url:
            self.base_url = base_url
        elif os.getenv("OPENAI_API_BASE"):
            self.base_url = os.getenv("OPENAI_API_BASE")
        elif os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            # 如果只有 OPENROUTER_API_KEY，默认使用 OpenRouter
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            # 默认使用 OpenAI（None）
            self.base_url = None
        
        # 创建 client
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return f"LLMAgent({self.model})"
    
    def llm_call(
        self, 
        system_content: str, 
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        """
        单轮 LLM 调用
        
        Args:
            system_content: System message（定义角色和约束）
            user_prompt: User message（具体任务）
            temperature: 温度参数（默认 0.0 确保确定性输出）
            max_tokens: 最大输出 token 数
            
        Returns:
            LLM 响应文本
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
        
        return response.choices[0].message.content
    
    def __repr__(self) -> str:
        base_info = f", base_url={self.base_url}" if self.base_url else ""
        return f"LLMAgent(model={self.model}{base_info})"
