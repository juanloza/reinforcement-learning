import numpy as np
from .base_replay_buffer import BaseReplayBuffer
from .replay_buffer_factory import register_buffer

@register_buffer('standard', 'standard-buffer', 'standardbuffer')
class PrioritizedReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity)
        self._buffer = []
        self._position = 0  # Inicializamos self.position aquí
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities:np.ndarray[np.float32] = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated, truncated):
        max_priority = self.priorities.max() if self._buffer else 1.0
        self.priorities[self._position] = max_priority

        element = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "terminated": terminated,
            "truncated": truncated
        }

        if len(self._buffer) < self.capacity:
            self._buffer.append(element)
        else:
            self._buffer[self._position] = element
        
        max_priority = self.priorities.max() if self.buffer else 1.0
        self.priorities[self._position] = max_priority
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("No samples in buffer.")

        priorities = self.priorities[:len(self._buffer)] ** self.alpha
        probabilities:np.ndarray[np.float32] = (priorities / priorities.sum())

        indices = np.random.choice(len(self._buffer), batch_size, p=probabilities)
        samples = [self._buffer[idx] for idx in indices]

        # Calcular los pesos de importancia
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        weights:np.ndarray[np.float32] = (len(self._buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices, errors, offset=0.1):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + offset) ** self.alpha