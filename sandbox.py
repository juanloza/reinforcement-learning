import numpy as np, gymnasium as gym
from evaluator.evaluator import Evaluator
from trainer.trainer import Trainer
from utils import Config
from model import MPLModel
from replay_buffer import ReplayBufferFactory, BaseReplayBuffer
from policy import PolicyFactory
from agent import DQNAgent
from keras import optimizers, metrics, losses, models
from experiment import ExperimentRunner

def running_old():
    # config
    config_path = './config/dqn_cartpole.yaml'
    config:dict = Config.load_config(config_path)

    # environment
    env_config = Config.get_config_vars(config, 'environment')
    env_name = env_config.pop("name")
    env:gym.Env = gym.make(env_name, **env_config)
    video_folder = "./data"
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: episode_id % 1 == 0,) # Guardar video cada 10 episodios
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # model
    model_config = Config.get_config_vars(config, 'model')
    model = MPLModel.create_model(input_shape, num_actions, model_config)

    # metricas
    metrics_config:dict = config.get('metrics', [])
    metrics_array = []
    for metric_config in metrics_config:  # Handle missing 'metrics' key
        metric_class_name = Config.get_config_vars(metric_config, 'class')
        metric_class = getattr(metrics, metric_class_name, None)  # Get metric class from tf.keras.metrics
        if metric_class:
            metric_params = Config.get_config_vars(metric_config, 'params')            
            metric_instance = metric_class(**metric_params)
            metrics_array.append(metric_instance)
        else:
            print(f"Warning: Metric class '{metric_class_name}' not found.")

    # compile
    model = models.load_model('cartpole-dqn.keras')
    # model.compile(loss="mse", optimizer=optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.00025, epsilon=0.01), metrics=metrics_array)
    model.summary()


    # policy
    policy_config = Config.get_config_vars(config, "policy")
    policy = PolicyFactory.create_policy(**policy_config)

    # replay buffer
    replay_config = Config.get_config_vars(config, "replay-buffer")
    replay:BaseReplayBuffer = ReplayBufferFactory.create_buffer(**replay_config)

    # agent
    agent_config = Config.get_config_vars(config, "agent")
    agent = DQNAgent(env, model, policy, replay, agent_config)

    # Crear instancias de Trainer y Evaluator
    trainer = Trainer(agent, num_episodes=1000, config=config)
    evaluator = Evaluator(agent, num_episodes=5)

    # Entrenar el agente
    trainer.train()

    # evaluar el modelo
    evaluator.evaluate()

def running_new():
    runner = ExperimentRunner('config/dqn_cartpole.yaml')
    runner.run()

running_new()