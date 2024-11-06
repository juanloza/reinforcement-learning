from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, num_actions, config:dict):
        self.num_actions = num_actions
        self.config = config

    @abstractmethod
    def select_action(self, predict_fn=None, **predict_args):
        pass

    def update(self):
        pass  # para políticas que necesitan actualizarse (como epsilon-greedy)