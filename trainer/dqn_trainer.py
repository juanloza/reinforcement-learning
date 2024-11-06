import numpy as np
from .base_trainer import BaseTrainer

class DQNTrainer(BaseTrainer):
    def run_episode(self):
        episode_metrics = {
            'reward': [],
            'epsilon': []
        }
        state, _ = self.agent.env.reset()
        done = False
        step = 0

        while not done:
            step += 1
            action = self.agent.select_action(np.reshape(state, [1, *self.agent.input_shape]))
            next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
            done = terminated or truncated

            # Guardar en buffer de experiencia
            self.agent.remember(state, action, reward, next_state, terminated, truncated)
            state = next_state

            episode_metrics['reward'].append(reward)
            episode_metrics['epsilon'].append(self.agent.policy.epsilon)

            # Ejecutar `replay` cada cierto número de pasos
            history = self.agent.replay()
            if history is not None:
                for metric_name, metric_value in history.history.items():
                    if metric_name not in episode_metrics:
                        episode_metrics[metric_name] = []
                    episode_metrics[metric_name].append(metric_value[0])

        return episode_metrics
