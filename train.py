import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import torch

# --- Importaciones de Stable Baselines 3 ---
from stable_baselines3 import PPO, SAC, TD3, DDPG
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecTransposeImage, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# --- Importaciones para la lectura de logs ---
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# --- Configuración de estilo para las gráficas ---
sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", 5)

# Configuración de hilos de PyTorch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# Evitar el conflicto de múltiples runtimes de OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RLExperimentManager:
    """
    Clase para gestionar un ciclo completo de experimentos de Reinforcement Learning:
    - Configuración de directorios.
    - Entrenamiento de agentes.
    - Generación de gráficas de aprendizaje.
    - Grabación de vídeos de los agentes.
    - Pruebas de robustez.
    - Análisis de resultados como los tiempos de entrenamiento.
    """
    def __init__(self, experiment_name="TFM_RL", n_envs=8, n_eval_envs=4):
        self.experiment_name = experiment_name
        self.n_envs = n_envs
        self.n_eval_envs = n_eval_envs
        self.base_dir = f"experiments/{experiment_name}"
        self.algorithms = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG, 'TRPO': TRPO}
        self._setup_directories()
        print(f"✅ Gestor de experimentos '{experiment_name}' inicializado en GPU con {n_envs} entornos paralelos.")

    def _setup_directories(self):
        dirs = ['models', 'videos', 'plots', 'results', 'logs']
        for d in dirs:
            os.makedirs(f"{self.base_dir}/{d}", exist_ok=True)
        os.makedirs(f"{self.base_dir}/plots/training_times", exist_ok=True)

    def create_vec_env(self, env_name, seed=0, n_envs=None, eval=False):
        n = n_envs or (self.n_eval_envs if eval else self.n_envs)
        def make_env(rank):
            def _init():
                env = gym.make(env_name)
                # Solo en el env principal mantenemos Monitor
                env = Monitor(env) if rank == 0 else env
                env.reset(seed=seed + rank)
                return env
            return _init

        envs = [make_env(i) for i in range(n)]
        vec_env = SubprocVecEnv(envs)
        if "CarRacing" in env_name:
            vec_env = VecTransposeImage(vec_env)
        return vec_env


    def train_agent(self, env_name, algo_name, total_timesteps=1_000_000, seed=0):
        print(f"\n🚀 Entrenando {algo_name} en {env_name} por {total_timesteps} timesteps con {self.n_envs} envs paralelos")
        env = self.create_vec_env(env_name, seed)

        # Política y dispositivo
        policy = "CnnPolicy" if "CarRacing" in env_name else "MlpPolicy"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Política: {policy} | 🔌 Dispositivo: {device}")

        # Parámetros base SIN TensorBoard logging para máxima velocidad
        base_kwargs = dict(
            policy=policy,
            env=env,
            verbose=0,
            seed=seed,
            device=device,
        )

        n_steps_env = min(256, max(1, total_timesteps // self.n_envs))
        if algo_name == "PPO":
            algo_kwargs = dict(
                **base_kwargs,
                n_steps=n_steps_env,               # pasos por entorno
                batch_size=64 * self.n_envs,
                n_epochs=10,
            )
        elif algo_name == "TRPO":
            algo_kwargs = dict(
                **base_kwargs,
                n_steps=n_steps_env,               # igual en TRPO
            )
        else:
            # Off-policy
            n_actions = env.action_space.shape[-1]
            action_noise = None
            if algo_name in ["TD3", "DDPG"]:
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            algo_kwargs = dict(
                **base_kwargs,
                buffer_size=200_000,
                batch_size=256,
                learning_starts=1_000,
                train_freq=1,
                gradient_steps=1,
            )
            if action_noise is not None:
                algo_kwargs["action_noise"] = action_noise

        model = self.algorithms[algo_name](**algo_kwargs)

        # Entrenamiento puro sin callbacks
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Cerrar entorno paralelo
        env.close()

        # Guardar modelo final
        final_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final"
        model.save(final_path)
        print(f"✅ Modelo final guardado en: {final_path}.zip")

        # Evaluación final manual de n_eval_envs episodios
        results_dir = os.path.join(self.base_dir, "results", f"{env_name}_{algo_name}")
        os.makedirs(results_dir, exist_ok=True)

        eval_env_clean = Monitor(gym.make(env_name))
        rewards = []
        for _ in range(self.n_eval_envs):
            obs, _ = eval_env_clean.reset()
            done = False
            total_r = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env_clean.step(action)
                total_r += reward
                done = terminated or truncated
            rewards.append(total_r)
        eval_env_clean.close()

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        np.savez(
            os.path.join(results_dir, "evaluations.npz"),
            timesteps=np.array([total_timesteps]),
            results=np.array([[mean_reward]]),
            stds=np.array([[std_reward]])
        )
        print(f"   💾 Evaluación final guardada en: {results_dir}/evaluations.npz  (mean={mean_reward:.2f} ±{std_reward:.2f})")

        # Liberar GPU
        del model
        torch.cuda.empty_cache()

    def plot_learning_curve(self, env_name, algo_names):
        print(f"\n📊 Generando curva de aprendizaje para {env_name}...")
        plt.figure(figsize=(12, 7))

        for idx, algo_name in enumerate(algo_names):
            results_path = os.path.join(
                self.base_dir, "results", f"{env_name}_{algo_name}", "evaluations.npz"
            )
            if not os.path.exists(results_path):
                print(f"   - Aviso: No se encontró el archivo de evaluación para {algo_name}. Se omitirá.")
                continue
            data = np.load(results_path)
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)

            plt.plot(timesteps, mean_rewards, label=algo_name, color=COLORS[idx])
            plt.fill_between(
                timesteps,
                mean_rewards - std_rewards,
                mean_rewards + std_rewards,
                alpha=0.2,
                color=COLORS[idx]
            )

        plt.title(f"Curva de Aprendizaje en {env_name}")
        plt.xlabel("Timesteps")
        plt.ylabel("Recompensa Media de Evaluación (±std)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        save_path = os.path.join(self.base_dir, "plots", f"{env_name}_learning_curve.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   💾 Gráfica guardada en: {save_path}")

    def record_agent_video(self, env_name, algo_name, num_episodes=3):
        """
        Graba un vídeo del agente entrenado interactuando con el entorno.
        
        Args:
            env_name (str): Nombre del entorno.
            algo_name (str): Nombre del algoritmo.
            num_episodes (int): Número de episodios a grabar.
        """
        print(f"📹 Grabando vídeo de {algo_name} en {env_name}...")
        model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_best_model.zip"
        if not os.path.exists(model_path):
             model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final.zip"
             if not os.path.exists(model_path):
                print(f"   - Error: No se encontró el modelo en {model_path}.")
                return

        model = self.algorithms[algo_name].load(model_path)
        video_dir = f"{self.base_dir}/videos/{env_name}_{algo_name}"
        
        # Envolver el entorno para grabar vídeo
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix=f"{algo_name}-episode")

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()
        print(f"   ✅ Vídeos guardados en: {video_dir}")

    def get_training_times_from_logs(self, envs, algos):
        """
        Lee los logs de TensorBoard y devuelve un diccionario con los tiempos de entrenamiento (en segundos).
        Este método calcula la duración usando las marcas de tiempo ('wall_time') de los eventos,
        lo que lo hace muy robusto.
        """
        print("\n⏱️  --- Iniciando lectura de tiempos de entrenamiento desde los logs --- ⏱️")
        base_log_dir = f"{self.base_dir}/logs"
        training_times = {}

        for env_name in envs:
            training_times[env_name] = {}
            for algo_name in algos:
                experiment_log_dir = os.path.join(base_log_dir, f"{env_name}_{algo_name}")
                # El feedback detallado se ha omitido para no saturar, pero la lógica es la misma.
                if not os.path.exists(experiment_log_dir): continue
                run_dirs = [d for d in os.listdir(experiment_log_dir) if os.path.isdir(os.path.join(experiment_log_dir, d))]
                if not run_dirs: continue
                run_dir_path = os.path.join(experiment_log_dir, run_dirs[0])
                event_files = glob.glob(os.path.join(run_dir_path, "events.out.tfevents.*"))
                if not event_files: continue
                
                try:
                    ea = EventAccumulator(event_files[0]).Reload()
                    scalar_tags = ea.Tags()['scalars']
                    if not scalar_tags: continue
                    events = ea.Scalars(scalar_tags[0])
                    if len(events) > 1:
                        duration = events[-1].wall_time - events[0].wall_time
                        training_times[env_name][algo_name] = duration
                except Exception:
                    continue
        
        print("✅ Lectura de tiempos completada.")
        return training_times

    def save_training_times(self, training_times):
        """
        Para cada entorno, guarda una gráfica de barras y una tabla LaTeX con los tiempos de entrenamiento.
        """
        print("\n📊 Generando gráficas de barras por entorno para los tiempos de entrenamiento...")
        output_dir = f"{self.base_dir}/plots/training_times"

        for env_name, algos_times in training_times.items():
            if not algos_times:
                print(f"   - Aviso: No hay datos de tiempo para '{env_name}'. Se omitirá.")
                continue
    
            algos = list(algos_times.keys())
            # Convertir a minutos para mejor visualización
            times_in_minutes = [t / 60 for t in algos_times.values()] 
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(algos, times_in_minutes, color=COLORS[:len(algos)])
            plt.ylabel("Tiempo de Entrenamiento (minutos)")
            plt.title(f"Tiempos de Entrenamiento en {env_name}")
            # Añadir valor encima de cada barra
            plt.bar_label(bars, fmt='%.2f')
            plt.tight_layout()
            
            save_path = f"{output_dir}/{env_name}_training_times.png"
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"   💾 Gráfica para {env_name} guardada en: {save_path}")

    def save_summary_table(self, training_times, envs, algos):
        """
        Crea y guarda una tabla resumen con los tiempos de entrenamiento (algoritmos vs entornos).
        La guarda en formato CSV y LaTeX.
        """
        print("\n📝 Generando tabla resumen de tiempos de entrenamiento...")
        output_dir = f"{self.base_dir}/plots/training_times"

        # Crear un DataFrame de pandas a partir del diccionario anidado
        df = pd.DataFrame(training_times).reindex(index=algos, columns=envs)
        
        # Convertir todos los tiempos a minutos y formatear
        df_minutes = df.map(lambda x: x / 60 if pd.notna(x) else np.nan)
        
        # Guardar como CSV
        csv_path = os.path.join(output_dir, "summary_training_times.csv")
        df_minutes.to_csv(csv_path, float_format="%.2f")
        print(f"   💾 Tabla resumen guardada en formato CSV en: {csv_path}")

        # Guardar como LaTeX
        latex_path = os.path.join(output_dir, "summary_training_times.tex")
        latex_string = df_minutes.to_latex(
            float_format="%.2f",
            na_rep="N/A", # Representación para valores no disponibles
            caption="Tabla Resumen de Tiempos de Entrenamiento (minutos)",
            label="tab:training_times",
            escape=False
        )
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_string)
        print(f"   💾 Tabla resumen guardada en formato LaTeX en: {latex_path}")


    def robustness_test(self, env_name, algo_name, noise_levels=[0.0, 0.1, 0.2], episodes=20):
        #Prueba la robustez añadiendo ruido gaussiano guardando medias y desviaciones.
        model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final.zip"
        if not os.path.exists(model_path):
            print(f"Modelo {model_path} no encontrado. Se omite robustez.")
            return

        # 1) Cargo el modelo en CPU para liberar GPU
        model = self.algorithms[algo_name].load(model_path, device="cpu")

        # 2) Entorno sencillo
        env = gym.make(env_name)
        summary = []
        print(f"🔍 Robustez: {algo_name} en {env_name}")

        for noise in noise_levels:
            rews = []
            for ep in range(episodes):
                obs, _ = env.reset()
                done = False
                total_r = 0.0
                while not done:
                    noisy_obs = obs + np.random.normal(0, noise, size=obs.shape)
                    action, _ = model.predict(noisy_obs, deterministic=True)
                    obs, r, terminated, truncated, _ = env.step(action)
                    total_r += r
                    done = terminated or truncated
                rews.append(total_r)
                if (ep+1) % max(1, episodes//5) == 0:
                    print(f"  ruido={noise:.2f} ep {ep+1}/{episodes}: r={total_r:.1f}")
            m, s = np.mean(rews), np.std(rews)
            summary.append((noise, m, s))
            print(f" • ruido={noise:.2f}: media={m:.1f} ±{s:.1f}")

        # 3) Gráfica
        df = pd.DataFrame(summary, columns=["noise", "mean_reward", "std_reward"])
        plt.figure(figsize=(8,5))
        plt.errorbar(df["noise"], df["mean_reward"], yerr=df["std_reward"], fmt="o-")
        plt.title(f"Robustez de {algo_name} en {env_name}")
        plt.xlabel("Ruido gaussiano σ")
        plt.ylabel("Recompensa media")
        plt.grid(True, linestyle="--", linewidth=0.5)
        out = f"{self.base_dir}/plots/{env_name}_{algo_name}_robustness.png"
        plt.savefig(out, dpi=300); plt.close()
        print(f"   ✅ Guardado robustez en: {out}")

        # 4) Limpieza
        env.close()
        del model
        torch.cuda.empty_cache()


# -------------------- USO DEL SISTEMA --------------------
if __name__ == "__main__":
    experiment = RLExperimentManager("exp01")
    
    # --- Definición de Entornos y Algoritmos ---
    envs = [
        #"Pendulum-v1",
        #"MountainCarContinuous-v0",
        #"LunarLanderContinuous-v3",
        "CarRacing-v3",
        #"BipedalWalker-v3"
    ]
    algos = ["PPO", "SAC", "TD3", "DDPG", "TRPO"]
    
    # --- Configuración de Timesteps por Entorno ---
    timesteps_per_env = {
        #"Pendulum-v1": 100_000,
        # "MountainCarContinuous-v0": 100_000,
        # "LunarLanderContinuous-v3": 500_000,
         "CarRacing-v3": 1_000,
        #"BipedalWalker-v3": 3_000_000,
    }

    # --- SECCIÓN DE ENTRENAMIENTO (Comentada para ejecutar solo el análisis) ---
    # Descomentar para realizar un entrenamiento completo.
    
    print("\n\n--- INICIANDO FASE DE ENTRENAMIENTO COMPLETA ---")
    for env_name in envs:
        for algo_name in algos:
            experiment.train_agent(env_name, algo_name, total_timesteps=timesteps_per_env[env_name], seed=0)
            experiment.record_agent_video(env_name, algo_name)
        experiment.plot_learning_curve(env_name, algos)
    print("\n--- ✅ FASE DE ENTRENAMIENTO FINALIZADA ---")
    

    # --- SECCIÓN DE ANÁLISIS (Se ejecuta siempre) ---
    print("\n\n--- INICIANDO FASE DE ANÁLISIS DE RESULTADOS ---")
    
    # 1. Obtener y guardar tiempos de entrenamiento

    training_times = experiment.get_training_times_from_logs(envs, algos)
    experiment.save_training_times(training_times)
    experiment.save_summary_table(training_times, envs, algos)
    
    print("\n\n--- ✅ Proceso de análisis completado con éxito. ---")

    # --- SECCIÓN DE PRUEBAS DE ROBUSTEZ (Comentada para ejecutar solo el análisis) ---
    # Descomentar para realizar las pruebas de robustez
    
    print("\n\n--- INICIANDO FASE DE ANÁLISIS DE ROBUSTEZ ---")
    for env_name in envs:
        for algo_name in algos:
            experiment.robustness_test(env_name, algo_name, noise_levels=[0.2, 0.5], episodes=2) 
    print("\n\n--- ✅ Proceso de análisis de robustez completado con éxito. ---")
    
