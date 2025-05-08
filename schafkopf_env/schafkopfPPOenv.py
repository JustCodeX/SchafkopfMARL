#Datei-Verwaltung
import glob
import os
import time

#Environment und Operation
import supersuit as ss
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from env.schafkopf_env import SchafkopfEnv  # Passe den Importpfad ggf. an

def train(env_fn, steps: int = 10, seed: int = 0, **env_kwargs):
    """
    Trainiert ein PPO-Modell auf dem Schafkopf-Environment.

    Parameters:
        env_fn: Funktion oder Klasse, die das Environment erzeugt (hier: SchafkopfEnv).
        steps (int): Anzahl der Trainingstimestep.
        seed (int): Zufalls-Seed für Reproduzierbarkeit.
        env_kwargs: Zusätzliche Parameter für die Env-Erzeugung.

    Returns:
        None
    """
    # Erzeuge das Environment
    env = env_fn(**env_kwargs)

    env.reset()
    
    # Konvertiere das PettingZoo-Environment in einen vektorisierten Gym-Wrapper
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
    
    # Falls noch nicht geschehen, wird hier automatisch der korrekte 
    # (flache) Beobachtungsraum verwendet – vorausgesetzt, dein Env liefert
    # bereits flache Box-Observations (z.B. MlpPolicy).
    model = PPO(MlpPolicy, env, verbose=1, seed=seed)
    
    # Training
    model.learn(total_timesteps=steps)
    
    # Speichern des Modells (Dateiname beinhaltet den Zeitstempel)
    model.save(f"{env.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("Model has been saved.")
    
    env.close()


def evaluate(env_fn, num_games: int = 10, seed: int = 0, **env_kwargs):
    """
    Evaluiert ein trainiertes PPO-Modell auf dem Schafkopf-Environment.

    Parameters:
        env_fn: Funktion oder Klasse, die das Environment erzeugt.
        num_games (int): Anzahl der zu spielenden Spiele.
        seed (int): Zufalls-Seed.
        env_kwargs: Zusätzliche Parameter für die Env-Erzeugung.
        
    Returns:
        float: Durchschnittlicher Reward pro Agent über alle Spiele.
    """
    env = env_fn(**env_kwargs)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    try:
        latest_model = max(glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("No saved model found!")
        return
    model = PPO.load(latest_model)
    
    rewards = {agent: 0 for agent in env.possible_agents}
    
    for game in range(num_games):
        obs = env.reset(seed=seed + game)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info= env.step(action)
            # Summiere den Reward aller Agenten
            for agent, r in reward.items():
                rewards[agent] += r
    env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards)
    avg_reward_per_agent = {agent: rewards[agent] / num_games for agent in rewards}
    print("Average reward per agent:", avg_reward_per_agent)
    return avg_reward


if __name__ == "__main__":
    # Hier kannst du zusätzliche Parameter an dein Schafkopf-Env übergeben
    env_kwargs = {}  # Beispiel: {"max_cycles": 100} etc.

    # Training: Trainiere ein PPO-Modell
    train(SchafkopfEnv, steps=10, seed=None, **env_kwargs)
    
    # Evaluation: Teste das trainierte Modell über 10 Spiele
    evaluate(SchafkopfEnv, num_games=10, seed=0, **env_kwargs)
