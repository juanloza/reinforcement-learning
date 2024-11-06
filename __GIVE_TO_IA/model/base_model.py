import numpy as np
from abc import ABC, abstractmethod
from keras import Model

class BaseModel(ABC):
    @classmethod
    @abstractmethod
    def create_model(cls, input_shape, num_actions, config:dict) -> Model:
        raise NotImplementedError