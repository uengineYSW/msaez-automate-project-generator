from typing import Any, Dict
from pydantic import ConfigDict, Field

from .base import BaseModelWithItem

class UserInfoModel(BaseModelWithItem):
    uid: str
    model_config = ConfigDict(extra="allow")
class InformationModel(BaseModelWithItem):
    projectId: str
    model_config = ConfigDict(extra="allow")
class InputsModel(BaseModelWithItem):
    selectedDraftOptions: Any = None
    userInfo: UserInfoModel = None
    information: InformationModel = None
    preferedLanguage: str = "Korean"
    jobId: str = ""