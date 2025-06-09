import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym

# --- Importaciones de Stable Baselines 3 ---
from stable_baselines3 import PPO, SAC, TD3, DDPG
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

# --- Importaciones para la lectura de logs ---
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# --- Configuración de estilo para las gráficas ---
sns.set_theme(style="whitegrid", palette="husl")
COLORS = sns.color_palette("husl", 5)

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
    def __init__(self, experiment_name="TFM_RL"):
        """
        Inicializa el gestor de experimentos.
        
        Args:
            experiment_name (str): Nombre base para el directorio de experimentos.
        """
        self.experiment_name = experiment_name
        self.base_dir = f"experiments/{experiment_name}"
        self.algorithms = {
            'PPO': PPO,
            'SAC': SAC,
            'TD3': TD3,
            'DDPG': DDPG,
            'TRPO': TRPO
        }
        self._setup_directories()
        print(f"✅ Gestor de experimentos '{experiment_name}' inicializado. Los resultados se guardarán en '{self.base_dir}'")

    def _setup_directories(self):
        """Crea los subdirectorios necesarios para guardar los resultados del experimento."""
        print("🛠️  Configurando directorios...")
        dirs = ['models', 'videos', 'plots', 'results', 'logs']
        for d in dirs:
            os.makedirs(f"{self.base_dir}/{d}", exist_ok=True)
            os.makedirs(f"{self.base_dir}/plots/training_times", exist_ok=True) # Directorio específico para tiempos

    def create_env(self, env_name, seed=0):
        """
        Crea y envuelve un entorno de Gymnasium.
        
        Args:
            env_name (str): Nombre del entorno.
            seed (int): Semilla para la reproducibilidad.

        Returns:
            DummyVecEnv: El entorno vectorizado y listo para usar.
        """
        def make_env():
            env = gym.make(env_name)
            env = Monitor(env) # Monitor para registrar estadísticas como la recompensa
            env.reset(seed=seed)
            return env
        env = DummyVecEnv([make_env])
        # CarRacing necesita un manejo especial de las observaciones (imágenes)
        if "CarRacing" in env_name:
            env = VecTransposeImage(env)
        return env

    def train_agent(self, env_name, algo_name, total_timesteps=100_000, seed=0):
        """
        Entrena un agente usando un algoritmo específico en un entorno dado.
        
        Args:
            env_name (str): Nombre del entorno.
            algo_name (str): Nombre del algoritmo (ej. 'PPO').
            total_timesteps (int): Número de pasos de tiempo para el entrenamiento.
            seed (int): Semilla para la reproducibilidad.
        """
        print(f"\n🚀 === Iniciando entrenamiento: {algo_name} en {env_name} por {total_timesteps} timesteps === 🚀")
        # Crear entornos de entrenamiento y evaluación
        env = self.create_env(env_name, seed=seed)
        eval_env = self.create_env(env_name, seed=seed+42)

        # Seleccionar la política y el dispositivo adecuados
        policy = "CnnPolicy" if "CarRacing" in env_name else "MlpPolicy"
        device = "cuda" if "CarRacing" in env_name else "cpu"
        print(f"🧠 Política: {policy} | 🔌 Dispositivo: {device}")

        # Argumentos comunes para todos los modelos
        common_kwargs = dict(
            policy=policy,
            env=env,
            verbose=0, # Reducido a 0 para no saturar la consola, los logs de TB son suficientes
            tensorboard_log=f"{self.base_dir}/logs/{env_name}_{algo_name}",
            seed=seed,
            device=device
        )

        """
        # Argumentos específicos para algoritmos off-policy
        if algo_name in ["SAC", "TD3", "DDPG"]:
            common_kwargs["buffer_size"] = 200_000
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            common_kwargs["action_noise"] = action_noise
        """

        # Instanciar el modelo
        model_class = self.algorithms[algo_name]
        model = model_class(**common_kwargs)

        # Callback para guardar el mejor modelo durante la evaluación
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.base_dir}/models/{env_name}_{algo_name}",
            log_path=f"{self.base_dir}/results/{env_name}_{algo_name}",
            eval_freq=max(5000, total_timesteps // 20), # Evaluar más a menudo en entrenamientos cortos
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )

        # Iniciar el entrenamiento
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
        
        # Guardar el modelo final
        final_model_path = f"{self.base_dir}/models/{env_name}_{algo_name}_final"
        model.save(final_model_path)
        print(f"✅ Entrenamiento completado. Modelo final guardado en: {final_model_path}.zip")

    def plot_learning_curve(self, env_name, algo_names):
        """
        Genera y guarda una gráfica comparando las curvas de aprendizaje de varios algoritmos.
        
        Args:
            env_name (str): Nombre del entorno.
            algo_names (list): Lista de nombres de algoritmos a comparar.
        """
        print(f"\n📊 Generando curva de aprendizaje para {env_name}...")
        plt.figure(figsize=(12, 7))
        for idx, algo_name in enumerate(algo_names):
            log_path = f"{self.base_dir}/results/{env_name}_{algo_name}/evaluations.npz"
            if not os.path.exists(log_path):
                print(f"   - Aviso: No se encontró el archivo de evaluación para {algo_name}. Se omitirá.")
                continue
            
            data = np.load(log_path)
            timesteps = data['timesteps']
            results = data['results']
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            
            plt.plot(timesteps, mean_rewards, label=algo_name, color=COLORS[idx])
            plt.fill_between(timesteps, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.2, color=COLORS[idx])
            
        plt.title(f"Curva de Aprendizaje en {env_name}")
        plt.xlabel("Timesteps")
        plt.ylabel("Recompensa Media de Evaluación (±std)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        save_path = f"{self.base_dir}/plots/{env_name}_learning_curve.png"
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
        df_minutes = df.applymap(lambda x: x / 60 if pd.notna(x) else np.nan)
        
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
        plt.errorbar(df["noise"], df["mean_reward"], yerr=df["std_reward"], fmt="r--o")
        plt.title(f"Robustez de {algo_name} en {env_name}")
        plt.xlabel("Nivel de ruido gaussiano")
        plt.ylabel("Recompensa media")
        plt.tight_layout()
        plt.savefig(f"{self.base_dir}/plots/{env_name}_{algo_name}_robustness.png", dpi=300)
        plt.close()


# -------------------- USO DEL SISTEMA --------------------
if __name__ == "__main__":
    experiment = RLExperimentManager("exp01")
    
    # --- Definición de Entornos y Algoritmos ---
    envs = [
        "Pendulum-v1",
        "MountainCarContinuous-v0",
        "LunarLanderContinuous-v3",
        #"CarRacing-v3",
        #"BipedalWalker-v3"
    ]
    algos = ["PPO", "SAC", "TD3", "DDPG", "TRPO"]
    
    # --- Configuración de Timesteps por Entorno ---
    timesteps_per_env = {
        "Pendulum-v1": 100_000,
        "MountainCarContinuous-v0": 100_000,
        "LunarLanderContinuous-v3": 500_000,
        "CarRacing-v3": 3_000_000,
        "BipedalWalker-v3": 3_000_000,
    }

    # --- SECCIÓN DE ENTRENAMIENTO (Comentada para ejecutar solo el análisis) ---
    # Descomentar para realizar un entrenamiento completo.
    
    print("\n\n--- INICIANDO FASE DE ENTRENAMIENTO COMPLETA ---")
    for env_name in envs:
        total_timesteps = timesteps_per_env[env_name]
        for algo_name in algos:
            experiment.train_agent(env_name, algo_name, total_timesteps=total_timesteps, seed=0)
            experiment.record_agent_video(env_name, algo_name)
        experiment.plot_learning_curve(env_name, algos)
    print("\n--- ✅ FASE DE ENTRENAMIENTO FINALIZADA ---")
    

    # --- SECCIÓN DE ANÁLISIS (Se ejecuta siempre) ---
    print("\n\n--- INICIANDO FASE DE ANÁLISIS DE RESULTADOS ---")
    
    # 1. Obtener y guardar tiempos de entrenamiento
    training_times = experiment.get_training_times_from_logs(envs, algos)
    experiment.save_training_times(training_times)
    
    # 2. Guardar la tabla resumen
    experiment.save_summary_table(training_times, envs, algos)
    
    print("\n\n--- ✅ Proceso de análisis completado con éxito. ---")

    # --- SECCIÓN DE PRUEBAS DE ROBUSTEZ (Comentada para ejecutar solo el análisis) ---
    # Descomentar para realizar las pruebas de robustez
    
    print("\n\n--- INICIANDO FASE DE ANÁLISIS DE ROBUSTEZ ---")
    for env_name in envs:
            for algo_name in algos:
                experiment.robustness_test(env_name, algo_name, noise_levels=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], episodes=10)
    
    print("\n\n--- ✅ Proceso de análisis de robustez completado con éxito. ---")
    
