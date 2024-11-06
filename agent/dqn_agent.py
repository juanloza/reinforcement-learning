import random
import numpy as np
from keras import Model, models
from gymnasium import Env
from .base_agent import BaseAgent
from .agent_factory import register_agent
from replay_buffer import BaseReplayBuffer
from policy import BasePolicy
from utils import Config
from keras.api.callbacks import ModelCheckpoint, History, EarlyStopping


@register_agent
class DQNAgent(BaseAgent):
    def __init__(self, env:Env, model:Model, policy:BasePolicy, replay_buffer:BaseReplayBuffer, **config):
        super().__init__(env, model, policy, replay_buffer, **config)

        self.num_episodes = Config.get_config_vars(config, 'episodes')
        self.gamma = Config.get_config_vars(config, 'gamma')    # discount rate
        self.batch_size = Config.get_config_vars(config, 'batch-size')

        self.checkpoint_callback = ModelCheckpoint(
            filepath='./checkpoints/checkpoint_{epoch:02d}.keras',  # Ruta donde se guardarán los checkpoints
            save_freq='epoch',  # Frecuencia de guardado (en este caso, cada epoch)
            monitor='loss',  # Métrica a monitorizar (opcional)
            save_best_only=True,  # Guardar solo el mejor modelo (opcional)
            verbose=0  # Mostrar mensajes de progreso (opcional)
        )

        self.early_stopping = EarlyStopping()


    def replay(self):
        if len(self.replay_buffer) < self.train_start:
            return None
        # Randomly sample minibatch from the memory
        minibatch = self.replay_buffer.sample(min(len(self.replay_buffer), self.batch_size))

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        states = np.array([exp["state"] for exp in minibatch])
        actions = np.array([exp["action"] for exp in minibatch])
        rewards = np.array([exp["reward"] for exp in minibatch])
        next_states = np.array([exp["next_state"] for exp in minibatch])
        terminateds = np.array([exp["terminated"] for exp in minibatch])
        truncateds = np.array([exp["truncated"] for exp in minibatch])
        dones = np.logical_or(terminateds, truncateds) # New 'done' flag combining 'terminated' and 'truncated'

        # Obtener las predicciones de Q-values para todos los estados en el batch
        targets_final = self.model.predict_on_batch(states)
        next_targets = self.model.predict_on_batch(next_states)

        # Calcular los valores objetivo Q
        targets = rewards + self.gamma * np.amax(next_targets, axis=1) * (1 - dones) 
        
        # Actualizar los Q-values correspondientes a las acciones tomadas
        targets_final[np.arange(self.batch_size), actions] = targets

        # Entrenar el modelo
        # metrics = self.model.train_on_batch(states, targets_final)
        metrics:History = self.model.fit(states, targets_final, verbose=0, callbacks=[self.checkpoint_callback])

        return metrics