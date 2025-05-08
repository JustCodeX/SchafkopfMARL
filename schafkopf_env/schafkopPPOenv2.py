import glob
import os
import time
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from pettingzoo.utils.conversions import aec_to_parallel
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
import pettingzoo.utils


class CustomDictExtractor(nn.Module):
    """
    Custom Feature Extractor, der aus einem Beobachtungs-Dictionary
    ausschließlich den 'observation'-Teil extrahiert.

    Attributes:
        features_dim (int): Dimension der extrahierten Features.
        extractor (nn.Module): Identitäts-Transformation (kann bei Bedarf erweitert werden).
    """
    def __init__(self, observation_space: gym.spaces.Dict):
        """
        Initialisiert den CustomDictExtractor.

        Parameters:
            observation_space (gym.spaces.Dict): Beobachtungsraum, aus dem der 'observation'-Teil extrahiert wird.
        """
        super().__init__()
        self.features_dim = observation_space.spaces["observation"].shape[0]
        self.extractor = nn.Identity()

    def forward(self, observations):
        """
        Führt die Extraktion des 'observation'-Teils aus dem Eingabe-Dictionary durch.

        Parameters:
            observations (dict): Dictionary mit mindestens dem Schlüssel "observation".

        Returns:
            Tensor: Die extrahierten Beobachtungsdaten.
        """
        return self.extractor(observations["observation"])


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):
    """
    Wrapper, der sicherstellt, dass reset() und step() ein Dictionary liefern,
    welches Beobachtungen mit 'observation' und 'action_mask' beinhaltet.

    Dieser Wrapper konvertiert den Spieler (z.B. als int) in ein Spieler-Objekt
    und passt die Observation- und Action-Spaces entsprechend an.
    """
    def __init__(self, env):
        """
        Initialisiert den SB3ActionMaskWrapper.

        Parameters:
            env: Das zugrunde liegende PettingZoo-Environment.
        """
        pettingzoo.utils.BaseWrapper.__init__(self, env)
        gym.Env.__init__(self)
        self.action_space = self.env.global_action_space
        self.observation_space = self.env.global_observation_space

    def reset(self, seed=None, options=None):
        """
        Setzt das Environment zurück und initialisiert interne Variablen.

        Parameters:
            seed (optional): Zufalls-Seed für Reproduzierbarkeit.
            options (optional): Zusätzliche Optionen für den Reset.

        Returns:
            tuple: (Beobachtung, info) – Die initiale Beobachtung als Dictionary und ein leeres Info-Dict.
        """
        observations, info = self.env.reset(seed=seed, options=options)
        # Initialisiere _cumulative_rewards anhand der Spielerpositionen
        self.env._cumulative_rewards = {spieler.position: 0 for spieler in self.env.spieler}
        return self._get_obs(self.env.spielerAmZug), info

    def _get_obs(self, spieler):
        """
        Erzeugt die Beobachtung als Dictionary, basierend auf dem Spieler-Objekt.

        Parameters:
            spieler: Der Spieler oder dessen Position.

        Returns:
            dict: Dictionary mit Schlüsseln "observation" und "action_mask".
        """
        obs = self.env.observe(spieler)
        return {"observation": obs["observation"], "action_mask": obs["action_mask"]}

    def observe(self, spieler):
        """
        Gibt die reine Beobachtung des angegebenen Agenten zurück.

        Parameters:
            Spieler: Der Spieler oder dessen Position.

        Returns:
            np.array: Beobachtungsvektor.
        """
        return self._get_obs(spieler)["observation"]

    def action_mask(self):
        """
        Gibt die Aktionsmaske des aktuell ausgewählten Agenten zurück.

        Returns:
            np.array: Aktionsmaske.
        """
        return self._get_obs(self.env.spielerAmZug)["action_mask"]

    def step(self, action):
        """
        Führt einen Schritt im Environment aus.

        Parameters:
            action: Entweder ein einzelner Aktionswert oder ein Dictionary von Aktionen für alle Agenten.

        Returns:
            tuple: (Beobachtung, reward, terminated, truncated, info) für den aktuell aktiven Agenten.
        """
        if not isinstance(action, dict):
            actions = {spieler.position: action for spieler in self.env.spieler}
        else:
            actions = action
            print("alle spieler machen gleiche move")
        self.env.step(actions)
        return self.last()

    def last(self):
        """
        Gibt die letzte Beobachtung, den Reward sowie den Terminations- und Truncation-Status
        des aktuell aktiven Agenten zurück.

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """
        spieler = self.env.spielerAmZug
        obs = self.env.observe(spieler)
        reward = self.env.reward()[spieler.position]
        terminated = self.env.dones.get(spieler.position, False) if hasattr(self.env, "dones") else False
        truncated = self.env.truncations.get(spieler.position, False) if hasattr(self.env, "truncations") else False
        info = {}
        return ({"observation": obs["observation"], "action_mask": obs["action_mask"]},
                reward, terminated, truncated, info)

    @property
    def unwrapped(self):
        """
        Gibt das zugrunde liegende Environment ohne Wrapper zurück.

        Returns:
            Das unwrapped Environment.
        """
        return self.env.unwrapped

    def render(self):
        """
        Gibt den aktuellen Render-Output des Environments zurück.

        Returns:
            Render-Ergebnis des Environments.
        """
        return self.env.render()


def mask_fn(env):
    """
    Callback-Funktion zur Bereitstellung der Aktionsmaske aus dem Environment.

    Parameters:
        env: Das Environment, von dem die Aktionsmaske abgefragt wird.

    Returns:
        np.array: Die Aktionsmaske.
    """
    return env.action_mask()


def train_action_mask(env_fn, steps, seed=0, **env_kwargs):
    """
    Trainiert ein MaskablePPO-Modell für das Schafkopf-Environment,
    welches als Dictionary Beobachtungen (mit 'observation' und 'action_mask')
    liefert.

    Parameters:
        env_fn: Funktion oder Klasse zum Erzeugen des Environments.
        steps (int): Anzahl der Trainingsschritte.
        seed (int): Zufalls-Seed für Reproduzierbarkeit.
        env_kwargs: Zusätzliche Keyword-Argumente für das Environment.

    Speichert:
        Das trainierte Modell im Ordner "models" mit einem Dateinamen basierend
        auf dem Environment-Namen und einem Zeitstempel.
    """
    # Environment erzeugen (als AEC-Env)
    env = env_fn(**env_kwargs)
    print(f"Starting training on {env.metadata['name']} mit {len(env.spieler)} Agenten.")

    # Environment in den SB3ActionMaskWrapper wickeln
    env = SB3ActionMaskWrapper(env)
    #env.reset(seed=seed)

    # ActionMasker anwenden
    env = ActionMasker(env, mask_fn)

    # Policy-Parameter definieren, um den CustomDictExtractor zu verwenden
    policy_kwargs = dict(
        features_extractor_class=CustomDictExtractor,
    )

    model = MaskablePPO(MaskableActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    # Speichern des Modells im Zielordner "models"
    save_folder = "models"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{env.unwrapped.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}")
    model.save(save_path)

    print("Model has been saved in", save_path)
    print(f"Finished training on {env.unwrapped.metadata['name']} mit {len(env.unwrapped.spieler)} Agenten.\n")
    env.close()


def eval_action_mask(env_fn, num_games, render_mode=None, **env_kwargs):
    """
    Führt eine Evaluation des trainierten MaskablePPO-Modells im Schafkopf-Environment durch.

    Parameters:
        env_fn: Funktion oder Klasse zum Erzeugen des Environments.
        num_games (int): Anzahl der Spiele zur Evaluation.
        render_mode: Render-Modus (falls benötigt).
        env_kwargs: Zusätzliche Keyword-Argumente für das Environment.

    Returns:
        tuple: (round_rewards, total_rewards) – Ergebnisse der Evaluation.
    """
    env = env_fn(**env_kwargs)
    env = SB3ActionMaskWrapper(env)
    print(f"Starting evaluation: Alle {len(env.unwrapped.spieler)} Agenten nutzen die trainierte Policy.")

    # Suche nach dem neuesten Modell im Ordner "models"
    model_pattern = os.path.join("models", f"{env.unwrapped.metadata['name']}*")
    model_files = glob.glob(model_pattern)
    if not model_files:
        print("Kein Modell gefunden.")
        return
    latest_policy = max(model_files, key=os.path.getctime)
    model = MaskablePPO.load(latest_policy)

    scores = {spieler: 0 for spieler in env.unwrapped.spieler}
    total_rewards = {spieler: 0 for spieler in env.unwrapped.spieler}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        for spieler in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                for a in env.unwrapped.spieler:
                    total_rewards[a] += env.rewards.get(a, 0)
                round_rewards.append(env.rewards)
                break
            else:
                act, _states = model.predict(obs, deterministic=True)
            env.step(act)

    env.close()

    print("Evaluation abgeschlossen.")
    print("Round rewards:", round_rewards)
    print("Total rewards:", total_rewards)

    return round_rewards, total_rewards


if __name__ == "__main__":
    # Hier können zusätzliche Parameter an das Schafkopf-Environment übergeben werden.
    from env.schafkopf_env import SchafkopfEnv  # Sicherstellen, dass der Import korrekt ist
    env_fn = SchafkopfEnv
    env_kwargs = {}  # Beispiel: {"max_cycles": 100} etc.

    train_action_mask(env_fn, steps=8, seed=0, **env_kwargs)
    eval_action_mask(env_fn, 8, **env_kwargs)
