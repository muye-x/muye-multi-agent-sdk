"""确定性或外部系统适配场景的 Custom Agent。"""
from __future__ import annotations

from .base import BaseAgent


class CustomAgent(BaseAgent):
    """Custom 模式不额外增加扩展点，子类实现 BaseAgent.execute 即可。"""
