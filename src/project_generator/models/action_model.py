from pydantic import ConfigDict, Field
from typing import Dict, Any, Optional

from .base import BaseModelWithItem

class ActionModel(BaseModelWithItem):
    """
    이벤트 스토밍 액션을 표현하는 모델 클래스
    
    Attributes:
        objectType (str): 액션 대상 객체 타입 (BoundedContext, Aggregate, Event 등)
        type (Optional[str]): 액션 유형 (create, update, delete)
        ids (Dict[str, str]): 객체 식별자 정보 (boundedContextId, aggregateId 등)
        args (Dict[str, Any]): 액션 실행에 필요한 인자들
    """
    objectType: str
    type: Optional[str] = None
    ids: Dict[str, str] = Field(default_factory=dict)
    args: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow") 