from pydantic import BaseModel

class BaseModelWithItem(BaseModel):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return key in self.model_dump()
    
    def keys(self):
        return self.model_dump().keys()

    def items(self):
        return self.model_dump().items()

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default
            
    def set(self, key, value):
        setattr(self, key, value)
        return value