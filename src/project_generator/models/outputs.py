from datetime import datetime
from pydantic import ConfigDict, Field
from typing import Any, Dict, List, Optional

from .base import BaseModelWithItem

class LogModel(BaseModelWithItem):
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    level: str = ""
    message: str = ""
    model_config = ConfigDict(extra="allow")

class EsValueModel(BaseModelWithItem):
    elements: Dict[str, Any] = Field(default_factory=dict)
    relations: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")

class OutputsModel(BaseModelWithItem):
    esValue: EsValueModel = EsValueModel()
    isCompleted: bool = False
    isFailed: bool = False
    logs: List[LogModel] = Field(default_factory=list)
    totalProgressCount: int = 0
    currentProgressCount: int = 0
    lastCompletedRootGraphNode: Optional[str] = None
    lastCompletedSubGraphNode: Optional[str] = None