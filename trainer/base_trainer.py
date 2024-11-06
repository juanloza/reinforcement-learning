import numpy as np
from agent import BaseAgent

class BaseTrainer:
    def __init__(self, agent:BaseAgent, num_episodes, config):
        self.agent = agent
        self.num_episodes = num_episodes
        self.config = config

    def train(self):
        train_metrics = {}
        for episode in range(self.num_episodes):
            episode_metrics = self.run_episode()
            self.log_metrics(episode, episode_metrics, train_metrics)
            
            # Guardado condicional de modelo
            if self.should_save_model(episode):
                self.agent.save(f"{self.agent.env.unwrapped.spec.id}-{self.agent.__class__.__name__}.keras")
            
            # Plotting u otras acciones cada cierto número de episodios
            if episode % 100 == 0:
                self.plot_metrics(train_metrics)

    def run_episode(self):
        """Define un episodio de entrenamiento. Método para ser implementado en subclases."""
        raise NotImplementedError

    def log_metrics(self, episode, episode_metrics, train_metrics):
        """Actualiza métricas por episodio y las muestra en consola."""
        for metric_name, metric_values in episode_metrics.items():
            if metric_name not in train_metrics:
                train_metrics[metric_name] = []
            if metric_name == 'reward':
                train_metrics[metric_name].append(np.sum(metric_values))
            elif metric_name == 'epsilon':
                train_metrics[metric_name].append(np.amin(metric_values))
            else:
                train_metrics[metric_name].append(np.mean(metric_values))

        print(f"Episode: {episode}", end='')
        [print(f", {metric_name}: {value[-1]:.4f}", end='') for metric_name, value in train_metrics.items()]
        print('')

    def should_save_model(self, episode):
        """Condiciones para guardar el modelo. Método opcional para ser personalizado en subclases."""
        return episode % 100 == 0
    
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
