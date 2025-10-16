"""
State model for Legacy compatibility (used by utils/job_util.py)
Note: This is retained for backward compatibility but not actively used by UserStory Generator.
"""
from .base import BaseModelWithItem
from .inputs import InputsModel
from .outputs import OutputsModel

class State(BaseModelWithItem):
    """Legacy State model for job management compatibility"""
    inputs: InputsModel = InputsModel()
    outputs: OutputsModel = OutputsModel()