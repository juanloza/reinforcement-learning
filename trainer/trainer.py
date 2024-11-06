import numpy as np
from agent.base_agent import BaseAgent
from keras.api.callbacks import History
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, agent:BaseAgent, num_episodes: int, config):
        self.agent = agent
        self.num_episodes = num_episodes
        self.config:dict = config

        # graphs:
        #   - title: Score
        #     figsize: [20, 15]
        #     xlabel: "Episodes"
        #     ylabel: "Values"
        #     plots:
        #     - data: "loss"
        #         label: "Perdida"
        #     - data: "mae"
        #         label: "Mean absolute error"
        #     - data: "root_mse"
        #         label: "Root mean squared error"
        #     - data: "mape"
        #         label: "Mean absolute percentage error"
        #     - data: "r2_score"
        #         label: "R^2 score"
    def plot_metrics(self, train_metrics):
        return
        graphs_config = self.config.get("graphs",[])
        for graph_config in graphs_config:
            plt.figure(figsize=graph_config.get("figsize",(30, 20)))
            plot_config = graph_config.get("plots",[])
            for plot in plot_config:
                plt.plot(train_metrics[plot.get("data","")], label=plot.get("label",""))
            plt.title(graph_config.get("title",""))
            plt.xlabel(graph_config.get("xlabel",""))
            plt.ylabel(graph_config.get("ylabel",""))
            plt.legend()
            # plt.savefig(filename)
            # plt.show()


    def train(self):
        train_metrics = {}
        for episode in range(self.num_episodes):
            state, _ = self.agent.env.reset()
            done = False
            step = 0
            episode_metrics = {}
            while not done:
                step += 1
                action = self.agent.select_action(np.reshape(state, [1, *self.agent.input_shape]))
                next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
                done = terminated or truncated
                # if not done or step >= self.agent.env._max_episode_steps:
                #     reward = reward
                # else:
                #     reward = -100
                self.agent.remember(state, action, reward, next_state, terminated, truncated)
                state = next_state
                
                history:History = self.agent.replay()
                if history is not None:
                   for metric_name, metric_value in history.history.items():
                       if metric_name not in episode_metrics:
                           episode_metrics[metric_name] = []  # Initialize if not already present
                       episode_metrics[metric_name].append(metric_value[0])

            for metric_name, metric_value in episode_metrics.items():
                if metric_name not in train_metrics:
                    train_metrics[metric_name] = np.zeros(episode).tolist()  # Initialize if not already present
                train_metrics[metric_name].append(np.mean(episode_metrics[metric_name]))

            print(f"episode: {episode}/{self.num_episodes}, score: {step}, epsilon: {self.agent.policy.epsilon:.4}", end='')
            [print(f", {metric_name}: {metric_value[episode]:.4}", end='') for metric_name, metric_value in train_metrics.items()]
            print('')
            if step >= self.agent.env._max_episode_steps:
                print("Saving trained model as cartpole-dqn.keras")
                self.agent.save("cartpole-dqn.keras")
                return
            
            if episode > 0 and episode % 100 == 0:
                self.plot_metrics(train_metrics)
                