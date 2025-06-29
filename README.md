# Comparativa de algoritmos de aprendizaje por refuerzo en tareas de control autónomo

Repositorio del Trabajo Fin de Máster (TFM)  
Máster Universitario en Inteligencia Artificial  
Universidad Internacional de La Rioja (UNIR)  
Autores: Ángel Alepuz Jerez y Víctor Clemente Nieto  
Director: Óscar Fernández Mora  
Fecha: 10 de julio de 2025

---

## Descripción

Este repositorio contiene el código, experimentos y documentación asociados al TFM titulado **"Comparativa de algoritmos de aprendizaje por refuerzo en tareas de control autónomo"**. El objetivo principal es analizar y comparar el rendimiento de cinco algoritmos de aprendizaje por refuerzo profundo en distintos entornos de control continuo, evaluando su eficiencia, estabilidad y robustez bajo un marco experimental reproducible.

---

## Motivación y objetivos

El aprendizaje por refuerzo profundo (DRL) ha demostrado ser una herramienta poderosa para el desarrollo de sistemas autónomos en robótica, vehículos y control industrial. Sin embargo, la selección del algoritmo más adecuado para cada tarea no es trivial, ya que el rendimiento varía significativamente según el entorno y los recursos disponibles.

**Objetivo general:**  
Comparar de forma exhaustiva cinco algoritmos de DRL en cinco entornos de control autónomo, identificando fortalezas, debilidades y recomendaciones prácticas para su aplicación.

**Objetivos específicos:**
- Evaluar el rendimiento en términos de recompensa acumulada, estabilidad y eficiencia computacional.
- Analizar la robustez de los agentes ante ruido sensorial.
- Proporcionar pautas para la selección de algoritmos en función de la tarea y el entorno.

---

## Algoritmos evaluados

Se han seleccionado cinco algoritmos representativos del estado del arte, implementados mediante la librería [Stable Baselines 3](https://stable-baselines3.readthedocs.io/):

- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed Deep Deterministic Policy Gradient)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **TRPO** (Trust Region Policy Optimization)

---

## Entornos de simulación

Los experimentos se han realizado en cinco entornos continuos de la librería [Gymnasium](https://gymnasium.farama.org/), cubriendo distintos desafíos de control autónomo:

| Entorno                     | Descripción                                      | Observación    | Acción      |
|-----------------------------|--------------------------------------------------|----------------|-------------|
| Pendulum-v1                 | Péndulo invertido                                | Vector         | Continua    |
| MountainCarContinuous-v0    | Coche en colina                                  | Vector         | Continua    |
| LunarLanderContinuous-v3    | Aterrizaje de módulo lunar                       | Vector         | Continua    |
| CarRacing-v3                | Conducción autónoma en circuito visual           | Imagen (RGB)   | Continua    |
| BipedalWalker-v3            | Locomoción de robot bípedo                       | Vector         | Continua    |

---

## Metodología experimental

- **Diseño factorial completo:** Cada algoritmo se entrena y evalúa en cada entorno.
- **Configuración:** Uso de hiperparámetros por defecto de Stable Baselines 3 para garantizar comparabilidad y reproducibilidad.
- **Entrenamiento:** Número de pasos ajustado según la complejidad del entorno (desde 100.000 hasta 3 millones de timesteps).
- **Evaluación:** Recompensa media acumulada, desviación estándar, tiempo de entrenamiento y robustez ante ruido gaussiano en las observaciones.
- **Visualización:** Curvas de aprendizaje, tablas comparativas y vídeos de comportamiento.

---

## Requisitos y entorno de ejecución
- **Python:** >= 3.8
- **Hardware recomendado:**  
  - CPU multinúcleo  
  - GPU NVIDIA con soporte CUDA 
**Principales librerías:**
- stable-baselines3
- gymnasium
- torch / tensorflow
- numpy, pandas, matplotlib, seaborn, plotly

Instalación de dependencias:
pip install -r requirements.txt

---

## Instrucciones de uso
1. **Clonar el repositorio:**
git clone https://github.com/Victoritos/TFM.git
cd TFM
2. **Instalar dependencias:**
pip install -r requirements.txt

---

## Contacto

Para dudas, sugerencias o colaboración, contactar con los autores a través de los issues del repositorio:

- Ángel Alepuz Jerez 
- Víctor Clemente Nieto

---

**Palabras clave:**  
Aprendizaje por refuerzo profundo, comparativa de algoritmos, Gymnasium, Stable Baselines 3, control autónomo, robustez, reproducibilidad.


