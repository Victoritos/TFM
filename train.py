import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3, DDPG
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", 5)

class RLExperimentManager:
    def __init__(self, experiment_name="TFM_RL"):
        self.experiment_name = experiment_name
        self.base_dir = f"experiments/{experiment_name}"
        self._setup_directories()
        self.algorithms = {
            'PPO': PPO,
            'SAC': SAC,
            'TD3': TD3,
            'DDPG': DDPG,
            'TRPO': TRPO
        }
        self.results = {}

    def _setup_directories(self):
        dirs = ['models', 'videos', 'plots', 'results', 'logs', 'vec_normalize']
        for d in dirs:
            os.makedirs(f"{self.base_dir}/{d}", exist_ok=True)

    def create_env(self, env_name, training=True, seed=0):
        def make_env():
            env = gym.make(env_name)
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        env = DummyVecEnv([make_env])
        # Solo usar VecNormalize para entornos que NO sean de imágenes
        if "CarRacing" not in env_name:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, training=training)
        # Para entornos de imágenes, aplicar VecTransposeImage
        if "CarRacing" in env_name:
            env = VecTransposeImage(env)
        return env

    def train_agent(self, env_name, algo_name, total_timesteps=100_000, seed=0):
        print(f"\n=== Entrenando {algo_name} en {env_name} ===")
        env = self.create_env(env_name, training=True, seed=seed)
        eval_env = self.create_env(env_name, training=False, seed=seed+42)

        policy = "CnnPolicy" if "CarRacing" in env_name else "MlpPolicy"
        device = "cuda" if "CarRacing" in env_name else "cpu"

        common_kwargs = dict(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f"{self.base_dir}/logs/{env_name}_{algo_name}",
            seed=seed,
            device=device
        )

        if algo_name in ["SAC", "TD3", "DDPG"]:
            common_kwargs["buffer_size"] = 200_000

        model_class = self.algorithms[algo_name]
        model = model_class(**common_kwargs)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.base_dir}/models/{env_name}_{algo_name}",
            log_path=f"{self.base_dir}/results/{env_name}_{algo_name}",
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )

        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        model.save(f"{self.base_dir}/models/{env_name}_{algo_name}_final")
        # Guardar VecNormalize solo si se usó (no para CarRacing)
        if "CarRacing" not in env_name:
            env.save(f"{self.base_dir}/vec_normalize/{env_name}_{algo_name}.pkl")
        print(f"Modelo {algo_name} entrenado y guardado para {env_name}")

    def plot_learning_curve(self, env_name, algo_names):
        plt.figure(figsize=(12, 7))
        for idx, algo_name in enumerate(algo_names):
            log_path = f"{self.base_dir}/results/{env_name}_{algo_name}/evaluations.npz"
            if not os.path.exists(log_path):
                continue
            data = np.load(log_path)
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            plt.plot(timesteps, mean_rewards, label=algo_name, color=COLORS[idx])
            plt.fill_between(timesteps, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=COLORS[idx])
        plt.title(f"Curva de aprendizaje en {env_name}")
        plt.xlabel("Timestep")
        plt.ylabel("Recompensa media (±std)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.base_dir}/plots/{env_name}_learning_curve.png", dpi=300)
        plt.close()

    def record_agent_video(self, env_name, algo_name, num_episodes=3):
        model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final.zip"
        if not os.path.exists(model_path):
            print(f"Modelo {model_path} no encontrado.")
            return
        model = self.algorithms[algo_name].load(model_path)
        env = gym.make(env_name, render_mode="rgb_array")
        video_dir = f"{self.base_dir}/videos/{env_name}_{algo_name}"
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()


    def robustness_test(self, env_name, algo_name, noise_levels=[0.0, 0.1, 0.2], episodes=20):
        model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final.zip"
        if not os.path.exists(model_path):
            print(f"Modelo {model_path} no encontrado.")
            return
        model = self.algorithms[algo_name].load(model_path)
        env = gym.make(env_name)
        rewards = []
        for noise in noise_levels:
            rewards_noise = []
            for _ in range(episodes):
                obs, _ = env.reset()
                total_reward = 0
                terminated, truncated = False, False
                while not (terminated or truncated):
                    noisy_obs = obs + np.random.normal(0, noise, size=np.shape(obs))
                    action, _ = model.predict(noisy_obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                rewards_noise.append(total_reward)
            rewards.append((noise, np.mean(rewards_noise), np.std(rewards_noise)))
        # Plot
        df = pd.DataFrame(rewards, columns=["noise", "mean_reward", "std_reward"])
        plt.figure(figsize=(8, 5))
        plt.errorbar(df["noise"], df["mean_reward"], yerr=df["std_reward"], fmt='-o')
        plt.title(f"Robustez de {algo_name} en {env_name}")
        plt.xlabel("Nivel de ruido gaussiano")
        plt.ylabel("Recompensa media")
        plt.tight_layout()
        plt.savefig(f"{self.base_dir}/plots/{env_name}_{algo_name}_robustness.png", dpi=300)
        plt.close()

# -------------------- USO DEL SISTEMA --------------------

if __name__ == "__main__":
    experiment = RLExperimentManager("TFM_RL")
    envs = [
        "Pendulum-v1",
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v3",
        "CarRacing-v3",
        "BipedalWalker-v3"
    ]
    algos = ["PPO", "SAC", "TD3", "DDPG", "TRPO"]

    # Entrenamiento real
    for env_name in envs:
        for algo_name in algos:
            experiment.train_agent(env_name, algo_name, total_timesteps=100_000, seed=0)
        experiment.plot_learning_curve(env_name, algos)

    # Grabación de vídeos
    for env_name in envs:
        for algo_name in algos:
            experiment.record_agent_video(env_name, algo_name)

    # Pruebas de robustez
    for env_name in envs:
        for algo_name in algos:
            experiment.robustness_test(env_name, algo_name, noise_levels=[0.0, 0.1, 0.2, 0.5], episodes=10)
