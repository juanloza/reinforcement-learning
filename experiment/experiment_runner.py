import gymnasium as gym
from keras import models, optimizers, metrics, losses
from utils import Config
from model import ModelFactory
from policy import PolicyFactory, BasePolicy
from replay_buffer import ReplayBufferFactory, BaseReplayBuffer
from agent import BaseAgent, AgentFactory
from trainer import DQNTrainer

class ExperimentRunner:
    def __init__(self, config_path):
        self.config:dict = Config.load_config(config_path)
        self.setup()

    def setup(self):
        # environment
        # env_config = Config.get_config_vars(self.config, 'environment')
        env_config:dict = self.config.get('environment', {})
        env_name = env_config.pop("name")
        self.env:gym.Env = gym.make(env_name, **env_config)
        # video_folder = "./data"
        # env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda episode_id: episode_id % 1 == 0,) # Guardar video cada 10 episodios
        # env = gym.wrappers.RecordEpisodeStatistics(env)

        input_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n

        # model
        # model_config = Config.get_config_vars(self.config, 'model')
        model_config = self.config.get('model',{})
        model_config['input_shape'] = input_shape
        model_config['num_actions'] = num_actions
        self.model = ModelFactory.create_model(**model_config)
        # model = MPLModel.create_model(input_shape, num_actions, model_config)

        # metricas
        metrics_config:dict[dict] = self.config.get('metrics', [])
        metrics_array = []
        for metric_config in metrics_config:  # Handle missing 'metrics' key
            metric_class_name = metric_config.get('class','')
            metric_class = getattr(metrics, metric_class_name, None)  # Get metric class from tf.keras.metrics
            if metric_class:
                metric_params = metric_config.get('params',{})            
                metric_instance = metric_class(**metric_params)
                metrics_array.append(metric_instance)
            else:
                print(f"Warning: Metric class '{metric_class_name}' not found.")

        # compile
        self.model:models.Model = models.load_model('cartpole-dqn.keras')
        # model.compile(loss="mse", optimizer=optimizers.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        self.model.compile(loss=losses.MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=0.00025), metrics=metrics_array)
        self.model.summary()


        # policy
        # policy_config = Config.get_config_vars(self.config, "policy")
        policy_config = self.config.get("policy", {})
        self.policy = PolicyFactory.create_policy(**policy_config)

        # replay buffer
        replay_config = self.config.get("replay-buffer", {})
        self.replay_buffer:BaseReplayBuffer = ReplayBufferFactory.create_buffer(**replay_config)

        # agent
        agent_config = self.config.get("agent", {})
        agent_config['env'] = self.env
        agent_config['model'] = self.model
        agent_config['policy'] = self.policy
        agent_config['replay_buffer'] = self.replay_buffer
        # agent = DQNAgent(env, model, policy, replay, agent_config)
        self.agent = AgentFactory.create_agent(**agent_config)

        # Crear instancias de Trainer y Evaluator
        self.trainer = DQNTrainer(self.agent, num_episodes=1000, config=self.config)
        # evaluator = Evaluator(agent, num_episodes=5)

        
        # # Crear trainer según el tipo especificado en la configuración
        # trainer_type = self.config["trainer"]["type"]
        # if trainer_type == "DQNTrainer":
        #     self.trainer = DQNTrainer(agent, num_episodes=self.config["trainer"]["num_episodes"], config=self.config)
        # else:
        #     raise ValueError(f"Unknown trainer type: {trainer_type}")

        # # Instancia del evaluador, si es necesario
        # self.evaluator = Evaluator(agent, num_episodes=self.config["evaluator"]["num_episodes"])

    def run(self):
        self.trainer.train()
        # self.evaluator.evaluate()
        pass
