import os
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Configuración
environments = ['Pendulum-v1', 'MountainCarContinuous-v0', 'BipedalWalker-v3']
timesteps = 10_000          # Total timesteps por entrenamiento
eval_episodes = 10       # Episodios por evaluación
eval_freq = 2000          # Frecuencia de evaluación intermedia
output_dir = 'trained_models10000'

os.makedirs(output_dir, exist_ok=True)

# Algoritmos a probar
target_algorithms = {
    'PPO': PPO,
    'SAC': SAC,
    'DDPG': DDPG,
    'TD3': TD3,
    'TRPO': TRPO
}

# Métricas a guardar
results = {}

for env_id in environments:
    results[env_id] = {}
    print(f"\n### Entorno: {env_id} ###")
    # Envoltorio para logging
    env = Monitor(gym.make(env_id))

    for name, Algo in target_algorithms.items():
        print(f"--> Entrenando {name}")
        # Instanciar y entrenar
        if name in ['SAC', 'TD3', 'DDPG']:
            model = Algo('MlpPolicy', env, verbose=0, buffer_size=100_000)
        else:
            model = Algo('MlpPolicy', env, verbose=0)

        start_time = time.time()

        # Evaluaciones intermedias
        evals = []
        n_stages = timesteps // eval_freq
        for stage in range(1, n_stages + 1):
            model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
            mean_ret, std_ret = evaluate_policy(model, env, n_eval_episodes=eval_episodes, deterministic=True)
            evals.append((stage * eval_freq, mean_ret))
            print(f"  - After {stage*eval_freq} steps: mean_eval={mean_ret:.2f} ±{std_ret:.2f}")

        # Evaluación final
        n_mean, n_std = evaluate_policy(model, env, n_eval_episodes=eval_episodes, deterministic=True)
        duration = time.time() - start_time
        print(f"  Entrenamiento {name} completo en {duration:.1f}s | Final mean={n_mean:.2f} ±{n_std:.2f}\n")

        # Guardar modelo
        model_path = os.path.join(output_dir, f"{env_id}_{name}")
        model.save(model_path)

        # Gráfica de rendimiento
        ts, rewards = zip(*evals)
        plt.figure()
        plt.plot(ts, rewards, marker='o')
        plt.xlabel('Timesteps')
        plt.ylabel('Mean Eval Reward')
        plt.title(f'{env_id} - {name}')
        plt.tight_layout()
        curve_path = os.path.join(output_dir, f"{env_id}_{name}_curve.png")
        plt.savefig(curve_path)
        plt.close()

        # Almacenar métricas
        results[env_id][name] = {
            'eval_curve': evals,
            'train_time_s': duration,
            'final_mean_reward': n_mean,
            'final_std_reward': n_std
        }

# Comparaciones globales
for env_id, algos in results.items():
    # Curvas comparativas
    plt.figure()
    for name, data in algos.items():
        ts, rewards = zip(*data['eval_curve'])
        plt.plot(ts, rewards, label=name)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Eval Reward')
    plt.title(f'Comparativa: {env_id}')
    plt.legend()
    plt.tight_layout()
    comp_path = os.path.join(output_dir, f"{env_id}_comparativa.png")
    plt.savefig(comp_path)
    plt.close()

    # Tabla de métricas finales en consola
    print(f"\n### Resultados finales {env_id} ###")
    print("Algoritmo  | Tiempo(s) | Mean Reward ± Std")
    print("---------|-----------|------------------")
    for name, data in algos.items():
        print(f"{name:8} | {data['train_time_s']:9.1f} | {data['final_mean_reward']:.2f} ± {data['final_std_reward']:.2f}")

print("\nProceso completado. Revisa 'trained_models/' para modelos y gráficos.")
