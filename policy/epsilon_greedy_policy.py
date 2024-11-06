import random, numpy as np
from utils import Config
from .base_policy import BasePolicy
from .policy_factory import register_policy

@register_policy('epsilon', 'epsilon-greedy')
class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, num_actions, **config:dict):
        super().__init__(num_actions, config)
        self.epsilon = Config.get_config_vars(config, 'epsilon')
        self.epsilon_min = Config.get_config_vars(config, 'epsilon-min')
        self.epsilon_decay = Config.get_config_vars(config, 'epsilon-decay')

    def select_action(self, predict_fn=None, **predict_args):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            q_values = predict_fn(**predict_args)
            return np.argmax(q_values)

    def update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay