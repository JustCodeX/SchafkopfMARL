#!/usr/bin/env python3
import argparse
import datetime
import glob
import logging
import os
import random
import time
from collections import OrderedDict
from matplotlib import pyplot as plt
import pandas as pd
from typing import List, Tuple, Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import uvicorn
from fastapi import Request
import matplotlib.pyplot as plt


import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
import numpy as np
import torch as th
import torch.nn as nn
import tensorboard_logs  # Falls benötigt, sonst entfernen
from pettingzoo.utils.conversions import aec_to_parallel
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pettingzoo.utils

from env.schafkopf_env import SchafkopfEnv  # Projektinterner Import

# ========================
# Globale Hilfsfunktionen
# ========================

def index_to_tuple(index, n1=2, n2=3, n3=8):
    """
    Konvertiert einen Index in ein Aktions-Tupel.

    Args:
        index (int): Der diskrete Index.
        n1, n2, n3 (int, optional): Dimensionen (Defaults: 2, 3, 8).

    Returns:
        tuple: Das entsprechende Aktions-Tupel.
    """
    a = index // (n2 * n3)
    remainder = index % (n2 * n3)
    b = remainder // n3
    c = remainder % n3
    return (a, b, c)

def tuple_to_index(self, action_tuple, n1=2, n2=3, n3=8):
    """
    Konvertiert ein Aktions-Tupel in einen eindeutigen Index.

    Args:
        action_tuple (tuple): Aktions-Tupel.
        n1, n2, n3 (int, optional): Dimensionen (Defaults: 2, 3, 8).

    Returns:
        int: Der berechnete Index.
    """
    a, b, c = action_tuple
    return a * (n2 * n3) + b * n3 + c

def mask_fn(env):
    """
    Callback-Funktion zur Bereitstellung der Aktionsmaske aus dem Environment.

    Args:
        env: Das Environment.

    Returns:
        np.array: Die Aktionsmaske.
    """
    return env.action_mask()


# ========================
# Environment & Policy Klassen
# ========================

class CustomDictExtractor(nn.Module):
    """
    Custom Feature Extractor, der aus einem Beobachtungs-Dictionary
    ausschließlich den 'observation'-Teil extrahiert.

    Attributes:
        features_dim (int): Dimension der extrahierten Features.
        extractor (nn.Module): Identitäts-Transformation.
    """
    def __init__(self, observation_space: gym.spaces.Dict):
        """
        Initialisiert den CustomDictExtractor.

        Args:
            observation_space (gym.spaces.Dict): Beobachtungsraum, aus dem der 'observation'-Teil extrahiert wird.
        """
        super().__init__()
        self.features_dim = observation_space.spaces["observation"].shape[0]
        self.extractor = nn.Identity()

    def forward(self, observations):
        """
        Führt die Extraktion des 'observation'-Teils aus dem Eingabe-Dictionary durch.

        Args:
            observations (dict): Dictionary mit mindestens dem Schlüssel "observation".

        Returns:
            Tensor: Die extrahierten Beobachtungsdaten.
        """
        return self.extractor(observations["observation"])


class DebugMaskablePolicy(MaskableActorCriticPolicy):
    """
    Debug-Variante der MaskableActorCriticPolicy, die zusätzliche Debug-Ausgaben
    während der Vorwärtsausführung liefert.
    """
    def forward(self, obs, deterministic=False, action_masks=None):
        """
        Führt den Vorwärtsschritt der Policy durch und maskiert die Aktionslogits.

        Args:
            obs (dict): Beobachtungen.
            deterministic (bool, optional): Gibt an, ob deterministisch gewählt wird (Default: False).
            action_masks (optional): Aktionmasken, falls nicht in obs vorhanden.

        Returns:
            tuple: (actions, value, log_prob).
        """
        if action_masks is None:
            action_masks = obs.get("action_mask", None)

        print("action_mask: ", action_masks)

        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)

        print("logits: ", logits)

        if action_masks is not None:
            mask_tensor = th.as_tensor(action_masks, device=logits.device, dtype=th.bool)
            masked_logits = logits.masked_fill(~mask_tensor, -1e8)
        else:
            masked_logits = logits

        print("masked_logit: ", masked_logits)

        distribution = th.distributions.Categorical(logits=masked_logits)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        value = self.value_net(latent_vf)

        return actions, value, log_prob


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):
    """
    Wrapper, der das PettingZoo-Environment in ein Gym-Interface konvertiert und
    sicherstellt, dass reset() und step() ein Dictionary mit 'observation' und 'action_mask' liefern.
    """
    def __init__(self, env):
        """
        Initialisiert den SB3ActionMaskWrapper.

        Args:
            env: Das zugrunde liegende PettingZoo-Environment.
        """
        pettingzoo.utils.BaseWrapper.__init__(self, env)
        gym.Env.__init__(self)
        self.action_space = self.env.global_action_space
        self.observation_space = self.env.global_observation_space

    def reset(self, seed, options=None):
        """
        Setzt das Environment zurück und sammelt für alle Spieler die Observations.

        Args:
            seed: Zufalls-Seed.
            options (optional): Zusätzliche Optionen für den Reset.

        Returns:
            tuple: (observations, info) – Beobachtungen als Dictionary und Info-Dictionary.
        """
        observations, info = self.env.reset(seed=seed, options=options)
        self.env._cumulative_rewards = {spieler.position: 0 for spieler in self.env.spieler}
        all_obs = {}
        for sp in self.env.spieler:
            obs_sp = self.env.observe(sp)
            all_obs[sp.position] = {"observation": obs_sp["observation"],
                                    "action_mask": obs_sp["action_mask"]}
        return {"observations": all_obs}, info

    def step(self, action):
        """
        Führt einen Schritt im Environment aus.

        Args:
            action: Entweder ein einzelner Aktionswert (als int) oder ein Dictionary von Aktionen.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) für alle Spieler.
        """
        if not isinstance(action, dict):
            if isinstance(action, int):
                action = index_to_tuple(action)
            actions = {spieler.position: action for spieler in self.env.spieler}
        else:
            actions = {k: index_to_tuple(v) if isinstance(v, int) else v for k, v in action.items()}
        self.env.step(actions)
        return self.last()

    def last(self):
        """
        Sammelt die finale Observation, Rewards und Statusinformationen.

        Returns:
            tuple: (observations, rewards, done, truncated, info) für alle Spieler.
        """
        all_obs = {}
        for sp in self.env.spieler:
            obs_sp = self.env.observe(sp)
            final_mask = (np.full_like(obs_sp["action_mask"], False)
                          if "action_mask" in obs_sp else None)
            all_obs[sp.position] = {"observation": obs_sp["observation"],
                                    "action_mask": final_mask}
        all_rewards = self.env.reward()
        done = self.env.unwrapped.done()
        truncated = self.env.truncations if hasattr(self.env, "truncations") else {}
        info = {"all_observations": all_obs, "all_rewards": all_rewards, "end_state": True}
        return ({"observations": all_obs}, all_rewards, done, truncated, info)

    @property
    def unwrapped(self):
        """
        Gibt das zugrunde liegende, unwrapped Environment zurück.

        Returns:
            Environment: Das unwrapped Environment.
        """
        return self.env.unwrapped

    def render(self):
        """
        Führt die Render-Funktion des zugrunde liegenden Environments aus.

        Returns:
            Das Resultat von env.render().
        """
        return self.env.render()


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper, der das Environment auf einen einzelnen Spieler fokussiert.
    """
    def __init__(self, env, spieler_index):
        """
        Initialisiert den SingleAgentWrapper.

        Args:
            env: Das zu umhüllende Environment.
            spieler_index (int): Index des fokussierten Spielers.
        """
        super(SingleAgentWrapper, self).__init__(env)
        self.spieler_index = spieler_index
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        """
        Setzt das Environment zurück und gibt die Observation des fokussierten Spielers zurück.

        Args:
            **kwargs: Zusätzliche Argumente für reset().

        Returns:
            tuple: (agent_obs, info) für den fokussierten Spieler.
        """
        full_obs, info = self.env.reset(**kwargs)
        agent_obs = full_obs["observations"][self.spieler_index]
        return agent_obs, info

    def step(self, action):
        """
        Führt einen Schritt im Environment aus und extrahiert die Rückgabe für den fokussierten Spieler.

        Args:
            action: Die Aktion des fokussierten Spielers.

        Returns:
            tuple: (agent_obs, agent_reward, done, truncated, info).
        """
        full_obs, full_reward, agent_done, agent_truncated, info = self.env.step(action)
        agent_obs = full_obs["observations"][self.spieler_index]
        agent_reward = full_reward[self.spieler_index] if isinstance(full_reward, dict) else full_reward
        return agent_obs, agent_reward, agent_done, agent_truncated, info

    def action_mask(self):
        """
        Gibt die Aktionsmaske des fokussierten Spielers zurück.

        Returns:
            np.array: Die Aktionsmaske des fokussierten Spielers.
        """
        agent = next(sp for sp in self.env.unwrapped.spieler if sp.position == self.spieler_index)
        obs = self.env.unwrapped.observe(agent)
        return obs["action_mask"]


class TurnBasedActionBufferWrapper(gym.Wrapper):
    """
    Wrapper, der turnbasierend Aktionen puffert und gesammelt an das Environment übergibt.
    """
    def __init__(self, env, ppo_agent_index):
        """
        Initialisiert den TurnBasedActionBufferWrapper.

        Args:
            env: Das zu umhüllende Environment.
            ppo_agent_index (int): Der Index des PPO-Agenten.
        """
        super().__init__(env)
        self.ppo_agent_index = ppo_agent_index
        self.action_buffer = {}
        self.acting_agent_position = self.env.unwrapped.spielerAmZug.position

    def step(self, action):
        """
        Puffert die Aktion des aktuell agierenden Spielers und führt den Schritt aus,
        sobald alle Aktionen vorliegen.

        Args:
            action: Die vom aktuellen Spieler übergebene Aktion.

        Returns:
            tuple: (obs, reward, done, truncated, info) aus dem Environment.
        """
        if not self.action_buffer:
            self.acting_agent_position = self.env.unwrapped.spielerAmZug.position
            self.env.unwrapped.stich = {}
        if isinstance(action, int):
            flat_index = action
        else:
            flat_index = int(action.tolist())
        action_tuple = index_to_tuple(flat_index)
        self.action_buffer[self.acting_agent_position] = action_tuple
        self.env.unwrapped.stich[self.acting_agent_position] = action_tuple
        if self.acting_agent_position != self.ppo_agent_index:
            if self.acting_agent_position < 3:
                self.acting_agent_position += 1
            else:
                self.acting_agent_position = 0
            return self._dummy_step_output()
        for spieler in self.env.unwrapped.spieler:
            if spieler.position not in self.action_buffer:
                default_act = self.default_action(self.env.unwrapped, spieler, self.action_buffer)
                self.action_buffer[spieler.position] = default_act
        full_actions = self.action_buffer.copy()
        self.action_buffer.clear()
        self.env.unwrapped.stich = {}
        obs, reward, done, truncated, info = self.env.step(full_actions)
        self.acting_agent_position = self.env.unwrapped.spielerAmZug.position
        self.env.render()
        done = self.env.unwrapped.done()
        return obs, reward, done, truncated, info
    
    def reset(self, *, seed = None, options = None):
        returnValue = super().reset(seed=seed, options=options)
        self.acting_agent_position = self.env.unwrapped.spielerAmZug.position
        return returnValue

    def _dummy_step_output(self):
        """
        Erzeugt eine Dummy-Rückgabe, falls nicht alle Spieler agiert haben.

        Returns:
            tuple: (obs, 0.0, False, False, {}) als Dummy-Ausgabe.
        """
        agent = next(sp for sp in self.env.unwrapped.spieler if sp.position == self.ppo_agent_index)
        obs = self.env.unwrapped.observe(agent)
        return obs, 0.0, False, False, {}

    def default_action(self, env, spieler, actions):
        """
        Gibt eine Standardaktion zurück, falls keine Aktion vorliegt.

        Args:
            env: Das Environment.
            spieler: Das Spielerobjekt.
            actions: Bereits vorliegende Aktionen.

        Returns:
            tuple: Eine erlaubte Standardaktion.
        """
        possible = self.default_actions(env, spieler, actions)
        if not possible:
            return (0, 0, 0)
        chosen = random.choice(possible)
        if isinstance(chosen, np.ndarray):
            chosen = tuple(chosen.tolist())
        return chosen

    def default_actions(self, env, spieler, actions):
        """
        Ermittelt alle erlaubten Aktionen für den Spieler.

        Args:
            env: Das Environment.
            spieler: Das Spielerobjekt.
            actions: Bereits vorliegende Aktionen.

        Returns:
            list: Liste erlaubter Aktions-Tupel.
        """
        if env.phase == "spielartWahl":
            return env.möglicheSpielartWahl(spieler, actions)
        elif env.phase == "kartenSpielen":
            return env.möglicheKartenWahl(spieler, actions)
        else:
            return [(0, 0, 0)]

    def get_spieler_by_position(self, position):
        """
        Sucht den Spieler anhand der Positionsangabe.

        Args:
            position (int): Die Position des Spielers.

        Returns:
            Spieler oder None: Gefundener Spieler oder None, falls nicht vorhanden.
        """
        for spieler in self.env.unwrapped.spieler:
            if spieler.position == position:
                return spieler
        return None

    def action_mask(self):
        """
        Gibt die Aktionsmaske für den aktuell agierenden Spieler zurück.

        Returns:
            np.array: Die Aktionsmaske.
        """
        agent = self.get_spieler_by_position(self.acting_agent_position)
        obs = self.env.unwrapped.observe(agent)
        print("active_agent ", agent)
        return obs["action_mask"]


# ========================
# Trainings- und Evaluationsfunktionen
# ========================

def train_single_agent_models(env_fn, steps, seed, env_kwargs={}):
    """
    Trainiert für jeden der 4 Agenten ein separates MaskablePPO-Modell.

    Args:
        env_fn: Funktion bzw. Klasse zum Erzeugen des Environments.
        steps (int): Anzahl der Trainingstimesteps.
        seed (int): Zufalls-Seed.
        env_kwargs (dict, optional): Zusätzliche Parameter für das Environment.

    Returns:
        None
    """
    num_agents = 4
    policy_kwargs = dict(
        features_extractor_class=CustomDictExtractor,
    )

    for agent_index in range(num_agents):
        print(f"Training model for Agent {agent_index}")
        global_env = env_fn(**env_kwargs)
        global_env = SB3ActionMaskWrapper(global_env)
        env = SingleAgentWrapper(global_env, agent_index)
        env = TurnBasedActionBufferWrapper(env, ppo_agent_index=agent_index)
        env = ActionMasker(env, mask_fn)

        # Vortrainiertes Modell laden, falls vorhanden (optional)
        print("Loading pretrained Model: ", agent_index)
        pretrained_model = MaskablePPO.load(f"models/13thGenAgents/schafkopf_agent_{agent_index}")

        model = MaskablePPO(
            DebugMaskablePolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            seed=seed,
            ent_coef=0.005,
            tensorboard_log="tensorboard_logs/"
            # ulimit -n 4096 tensorboard --logdir=tensorboard_logs/
        )
        model.policy.load_state_dict(pretrained_model.policy.state_dict())
        model.learn(total_timesteps=steps)

        os.makedirs("models/Test", exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("models/Test", f"schafkopf_agent_{agent_index}_{current_time}")
        #model.save(save_path)
        print(f"Model for Agent {agent_index} saved at {save_path}")
        env.close()


def createActionMask(spieler, actions, env):
    """
    Erstellt eine Aktionsmaske für den gegebenen Spieler basierend auf erlaubten Aktionen.

    Args:
        spieler: Das Spielerobjekt.
        actions: Aktuelle, bereits gesammelte Aktionen.
        env: Das Environment.

    Returns:
        np.array: Die 48-dimensionale Aktionsmaske.
    """
    action_mask = np.full(48, False, dtype=bool)
    if env.phase == "spielartWahl":
        möglicheActions = env.möglicheSpielartWahl(spieler, actions)
    elif env.phase == "kartenSpielen":
        möglicheActions = env.möglicheKartenWahl(spieler, actions)
    else:
        möglicheActions = []
    if möglicheActions:
        for action_tuple in möglicheActions:
            index = env.tuple_to_index(action_tuple)
            action_mask[index] = True
    return action_mask

def evaluate_models(env_fn, num_episodes):
    """
    Lädt vier Modelle und lässt sie gegeneinander spielen.

    Args:
        env_fn (callable): Funktion, die ein neues Environment erzeugt.
        num_episodes (int): Anzahl der zu spielenden Episoden.

    Returns:
        dict: Dictionary, in dem für jeden Agenten eine Liste der gesammelten Rewards enthalten ist.
    """
    def load_latest_model(agent_index, folder):
        pattern = os.path.join(folder, f"schafkopf_agent_{agent_index}.zip")
        model_files = glob.glob(pattern)
        if not model_files:
            raise FileNotFoundError(f"Kein Modell für Agent {agent_index} gefunden.")
        latest_model_file = max(model_files, key=os.path.getmtime)
        return MaskablePPO.load(latest_model_file)

    num_agents = 4
    folder = "models/4thGenAgents"
    models = {agent: load_latest_model(agent, folder) for agent in range(num_agents)}
    results = {i: [[], []] for i in range(num_agents)}

    for ep in range(num_episodes):
        env = env_fn()
        obs, infos = env.reset()
        done = {i: False for i in range(num_agents)}
        total_rewards = {i: 0 for i in range(num_agents)}
        total_punkte = {i: 0 for i in range(num_agents)}

        while not all(done.values()):
            actions = {}
            punkte = {}
            for agent in obs:
                model = models[agent]
                spieler = next(sp for sp in env.unwrapped.spieler if sp.position == agent)
                print(f"Spieler {spieler.position}, Hand: {[str(karte) for karte in spieler.hand]}")
                action_mask = createActionMask(spieler, actions, env)
                action, _ = model.predict(obs[agent], action_masks=action_mask, deterministic=True)
                if not isinstance(action, (tuple, list)):
                    action = index_to_tuple(action)
                allowed_actions = []
                if env.phase == "spielartWahl":
                    allowed_actions = env.möglicheSpielartWahl(spieler, actions)
                elif env.phase == "kartenSpielen":
                    allowed_actions = env.möglicheKartenWahl(spieler, actions)
                if allowed_actions and action not in allowed_actions:
                    print(f"Agent {agent} predicted invalid action {action}, substituting default.")
                    action = allowed_actions[0]
                print("gewählte action: ", action)
                actions[agent] = action
            obs, rewards, done, truncated, infos = env.step(actions)
            for agent, rew in rewards.items():
                total_rewards[agent] += rew
            for spieler in env.unwrapped.spieler:
                total_punkte[spieler.position] = spieler.punkte
            print(f"Rewards im Step: {rewards}\n")
            env.render()
        print(f"Episode {ep} - Total Rewards: {total_rewards}")
        for agent in total_rewards:
            results[agent][0].append(total_rewards[agent])
            results[agent][1].append(total_punkte[agent])
        env.close()
    print(results)
    return results


def decode_agent_action(action, phase, env, spieler):
    """
    Wandelt die Agentenaktion in einen lesbaren String um,
    abhängig von der aktuellen Phase (Spielartwahl oder Kartenauswahl).

    Args:
        action: Die vom Agenten vorhergesagte Aktion (Index oder Tupel).
        phase: Aktuelle Phase im Environment.
        env: Das Environment.
        spieler: Das Spielerobjekt.

    Returns:
        str: Eine menschenlesbare Beschreibung der Aktion.
    """
    if phase == "spielartWahl":
        mapping = {0: "Passen", 1: "Rufspiel"}
        if isinstance(action, int):
            base = mapping.get(action, f"Unbekannt ({action})")
        elif isinstance(action, (tuple, list)):
            base = mapping.get(action[0], f"Unbekannt ({action[0]})")
            if base == "Rufspiel":
                ass_mapping = {0: "Alte", 1: "Blaue", 2: "Bumbs"}
                try:
                    ass_index = int(action[1])
                except (ValueError, TypeError):
                    ass_index = action[1]
                base += f" ({ass_mapping.get(ass_index, f'Unbekannt {ass_index}')})"
        else:
            base = str(action)
        return base
    elif phase == "kartenSpielen":
        idx = action[2]
        if idx < 0 or idx >= len(spieler.hand):
            return "Ungültiger Kartenindex"
        return str(spieler.hand[idx])
    else:
        return str(action)


def main_console(episodes: int, num_agents: int) -> None:
    """
    Konsolenmodus: Spiel zwischen agentengesteuerten und menschlichen Spielern.

    Datensammlung: Für jeden Trick werden gespeichert:
      - Episode, Tricknummer
      - Spielart (game_type) und Partner (falls in info)
      - Aktionen aller Spieler (decoded)
      - Rewards dieses Tricks und kumulierte Gesamt-Rewards aller Spieler

    Args:
        episodes (int): Anzahl der zu spielenden Episoden.
        num_agents (int): Anzahl der agentengesteuerten Spieler (0–4).
    """
    env = SchafkopfEnv()
    
    TOTAL_PLAYERS = 4
    actions: Dict[int, Union[int, Tuple[int, ...]]] = {}

    # frontend variablen
    current_player_cards: Dict[int, List[str]] = {}
    aktueller_stich: Dict = {}
    letzter_stich: Dict = {}
    aktueller_spieler_am_zug: int = -1
    spielerNamen: Dict = {}
    letzterGewinner: List[str] = []

    if not (0 <= num_agents <= TOTAL_PLAYERS):
        print(f"Ungültige Anzahl an Agenten: {num_agents} (muss zwischen 0 und {TOTAL_PLAYERS} liegen)")
        return

    # Starte FastAPI im Hintergrund-Thread
    app = FastAPI()

    # CORS Middleware für lokale Tests (Frontend-Verbindung)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/player/{player_id}/cards")
    async def get_player_cards(player_id: int):
        return {"player_id": player_id, "cards": current_player_cards.get(player_id, [])}

    @app.get("/trick")
    async def get_current_trick():
        return {
            "trick": [
                {"spieler": k, "karte": v}
                for k, v in (aktueller_stich.items())
            ] }
    
    @app.get("/last_trick")
    async def get_last_trick():
        return {
            "last_trick": [
                {"spieler": k, "karte": v}
                for k, v in (letzter_stich.items())
            ] }
    
    @app.get("/turn")
    async def get_current_turn():
        return {"spieler_am_zug": aktueller_spieler_am_zug}

    @app.post("/player/{player_id}/name")
    async def set_player_name(player_id: int, request: Request):
        data = await request.json()
        name = data.get("name", f"Spieler {player_id}")
        spielerNamen[player_id] = name
        print(f"\033[92m✅ Spielername gesetzt:\033[0m ID {player_id} → „{name}“")
        return {"success": True}

    @app.get("/player/{player_id}/name")
    async def get_name(player_id: int):
        return {"name": spielerNamen.get(player_id, f"Spieler {player_id}")}
    
    @app.get("/winner")
    async def get_winner():
        return {"winner": letzterGewinner}
    
    @app.get("/gerufenes_ass")
    async def get_gerufenes_ass():
        if env.gerufene:
            return {"ass": str(env.gerufene)}
        return {"ass": None}
    
    @app.get("/rollen")
    async def get_rollen():
        rollen = {}
        for p in env.spieler:
            if p.rolle == "Spielmacher":
                rollen[str(p.position)] = "Spielmacher"
            elif p.rolle != "Nicht-Spieler" and env.partnerAufgedeckt:
                rollen[str(p.position)] = str(env.gerufene)
            else:
                rollen[str(p.position)] = "Nicht-Spieler"
        return rollen

    @app.get("/partner_aufgedeckt")
    async def get_partner_aufgedeckt():
        return {"aufgedeckt": env.partnerAufgedeckt}
    
    @app.get("/player/{player_id}/legal_moves")
    async def get_legal_moves(player_id: int):
        if env.phase == "kartenSpielen":
            spieler = next(spieler for spieler in env.spieler if spieler.position == player_id)
            moeglich = env.möglicheKartenWahl(spieler, actions)
            if moeglich is None:
                return []
            return [str(karte) for karte in moeglich]
        else:
            return []

    def run_api():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="warning" 
        )

    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()

    # Datensammlung initialisieren
    records: list = []

    # Agentenmodelle laden
    models: Dict[int, MaskablePPO] = {}
    for idx in range(num_agents):
        model_path = os.path.join("models", "13_4thGenAgents", f"schafkopf_agent_{idx}.zip")
        if not os.path.isfile(model_path):
            print(f"Modell für Agent {idx} nicht gefunden: {model_path}")
            return
        models[idx] = MaskablePPO.load(model_path)
        print(f"Agent {idx} geladen aus {model_path}")

    try:
        for runde in range(1, episodes + 1):
            print(f"\n=== Runde {runde}/{episodes} startet ===\n")
            observations, info = env.reset()
            current_player_cards = {p.position: [str(card) for card in p.hand] for p in env.spieler}
            done_flags: Dict[int, bool] = {p.position: False for p in env.spieler}
            total_rewards: Dict[int, float] = {p.position: 0.0 for p in env.spieler}
            stich = 0
            partnerAufgedeckt = False

            while not all(done_flags.values()):
                actions = {}

                # Aktionen ermitteln
                for spieler in env.spieler:
                    spos = spieler.position
                    aktueller_spieler_am_zug = spos
                    if spos in models:
                        obs = observations[spos]
                        mask = createActionMask(spieler, actions, env)
                        raw_action, _ = models[spos].predict(obs, action_masks=mask, deterministic=True)
                        action = raw_action if isinstance(raw_action, (tuple, list)) else index_to_tuple(raw_action)
                        actions[spos] = action
                        decoded = decode_agent_action(action, env.phase, env, spieler)
                        aktueller_stich[spos] = decoded
                        time.sleep(random.uniform(2,4))
                        if env.phase =="spielartWahl":
                            env.angesagt = decoded
                        print(f"Agent {spos} wählt: {decoded}")
                    else:
                        if env.phase == "spielartWahl":
                            human_action = env.spielartWählen(spieler, actions)
                        elif env.phase == "kartenSpielen":
                            human_action = env.karteWählen(spieler, actions)
                        else:
                            raise RuntimeError(f"Unbekannte Phase: {env.phase}")
                        actions[spos] = human_action
                        decoded = decode_agent_action(human_action, env.phase, env, spieler)
                        aktueller_stich[spos] = decoded
                        time.sleep(2)
                        if env.phase =="spielartWahl":
                            env.angesagt = decoded
                        print(f"\nAgent {spos} (Mensch) wählt: {decoded} \n")

                # Datensatz anlegen
                record: Dict[str, Union[int, str, float]] = {
                    "runde": runde,
                    "stich": stich,
                    "game_type": env.spielart,
                }
                
                # Aktionen der Spieler
                for idx, pid in enumerate(actions.keys()): 
                    action = actions[pid]
                    decoded = decode_agent_action(
                        action,
                        env.phase,
                        env,
                        next(p for p in env.spieler if p.position == pid)
                    )
                    record[f"action_{idx}"] = f"{pid}:{decoded}"

                # Schritt ausführen
                observations, rewards, new_done, _, info = env.step(actions)
                env.render()
                stich += 1
                for spieler in env.spieler:
                    current_player_cards[spieler.position] = [str(card) for card in spieler.hand]
                letzter_stich.clear()
                letzter_stich.update(aktueller_stich)
                aktueller_stich.clear()
                
                # Partnerinfo einfügen
                partnerAufgedeckt = env.partnerAufgedeckt
                record[f"partnerAufgedeckt"] = partnerAufgedeckt
                for spieler in env.spieler:
                    spielPartner = ""
                    for partner in spieler.partner:
                        spielPartner += f"{partner.position}"
                    record[f"spieler{spieler.position}partner"] = spielPartner

                # Punkte einfügen
                for spieler in env.spieler:
                    spielerPunkte = spieler.punkte
                    record[f"punkte{spieler.position}"] = spielerPunkte

                # Rewards und kumulierte Totals
                for spos, rew in rewards.items():
                    total_rewards[spos] += rew
                    record[f"reward_{spos}"] = rew
                    record[f"total_{spos}"] = total_rewards[spos]

                records.append(record)
                done_flags.update(new_done)
                print(f"Step-Rewards: {rewards}\n")

            # Rundenende anzeigen
            env.render()
            letzter_stich.clear()

            #Gewinnerzeile
            if hasattr(env, "gewinnerTeam"):
                gewinner_namen = [spielerNamen.get(p.position, f"Spieler {p.position}") for p in env.gewinnerTeam]
                print(f"🏆 Gewinnerteam: {', '.join(gewinner_namen)}")

                # global speichern für API
                letzterGewinner = gewinner_namen

                # neue, leere Zeile nur mit dem Gewinnerteam
                gewinner_zeile = {col: "" for col in record.keys()}
                gewinner_zeile["runde"] = runde
                gewinner_zeile["stich"] = "🏆"
                gewinner_zeile["gewinner"] = ", ".join(gewinner_namen)
                records.append(gewinner_zeile)
            time.sleep(5) 

            # if "round_reward" in info:
            #     print(f"Runden-Rewards: {info['round_reward']}")
            # else:
            #     print(f"Episode {runde} beendet. Gesamte Rewards: {total_rewards}")

    except KeyboardInterrupt:
        print("Spiel durch Benutzer abgebrochen.")
    finally:
        env.close()

    # DataFrame erstellen und Spalten sortieren
    df = pd.DataFrame(records)
    # Basis-Spalten sicherstellen (runde, stich, game_type, partner)
    cols = (
        ['runde', 'stich']                        
        + ['game_type']                             
        + [f'action_{i}' for i in range(TOTAL_PLAYERS)]
        + [f'spieler{i}partner' for i in range(TOTAL_PLAYERS)]
        + ['partnerAufgedeckt']
        + [f'punkte{i}' for i in range(TOTAL_PLAYERS)]
        + [f'reward_{i}' for i in range(TOTAL_PLAYERS)]
        + [f'total_{i}' for i in range(TOTAL_PLAYERS)]
        + [f'gewinner']
    )
    df = df[cols]

    for i in range(TOTAL_PLAYERS):
        col = f'action_{i}'
        if col in df.columns:
            df[col] = df[col].apply(
                lambda eintrag: eintrag.replace(
                    f"{eintrag.split(':')[0]}:",
                    f"{spielerNamen.get(int(eintrag.split(':')[0]), f'Spieler {i}')}:" if ':' in eintrag else eintrag
                )
            )
    for i in range(TOTAL_PLAYERS):
        partner_col = f'spieler{i}partner'
        if partner_col in df.columns:
            df[partner_col] = df[partner_col].apply(
                lambda pstr: ','.join([spielerNamen.get(int(pid), f"Spieler {pid}") for pid in list(pstr)])
            )

    # Ergebnisse-Verzeichnis
    output_dir = "consoleResults/eval"
    os.makedirs(output_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV export
    csv_out = os.path.join(output_dir,f"schafkopf_results_{current_time}.csv")
    df.to_csv(csv_out, index=False)
    print(f"CSV gespeichert als '{csv_out}'")

    # Excel export mit formatierter Breite
    xlsx_out = os.path.join(output_dir,f"schafkopf_results_{current_time}.xlsx")
    with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        workbook = writer.book
        worksheet = writer.sheets["Results"]
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, max_len)
    print(f"Excel gespeichert als '{xlsx_out}'")

def plot_combined_rewards(results):
    """
    Zeichnet zwei Subplots in einem Fenster:
    - Oben: Total Reward pro Episode
    - Unten: Spieler-Punkte pro Episode (Y-Achse bis 120)
    results: dict {agent_index: (reward_list, points_list)}
    """
    # Episode-Zähler basierend auf Länge der Reward-Liste
    reward_list, _ = next(iter(results.values()))
    episodes = list(range(0, len(reward_list)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Linienplot für Rewards
    for agent, (rewards, _) in results.items():
        ax1.plot(episodes, rewards, marker='o', label=f"Agent {agent}")
    ax1.set_title("Total Reward pro Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True)

    # Linienplot für Punkte mit y-Limit 0–120
    for agent, (_, points) in results.items():
        ax2.plot(episodes, points, marker='x', label=f"Spieler {agent}")
    ax2.set_title("Spieler-Punkte pro Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Punkte")
    ax2.set_ylim(0, 120)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """
    Hauptfunktion: Parst die Kommandozeilenargumente und startet den gewünschten Modus
    (Training, Evaluation oder Consolensteuerung).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Schafkopf Agenten: Training, Evaluation oder Console-Modus")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "console"], required=True,
                        help="Modus: 'train' für Training, 'eval' für Evaluation, 'console' für Konsolensteuerung")
    parser.add_argument("--steps", type=int, default=1000, help="Anzahl der Trainingstimesteps bzw. Episoden im Console-Modus")
    parser.add_argument("--episodes", type=int, default=10, help="Anzahl der Evaluations-Episoden (nur für eval)")
    parser.add_argument("--agents", type=int, default=1, help="Anzahl der agent-gesteuerten Spieler (nur im Console-Modus)")
    args = parser.parse_args()

    if args.mode == "train":
        env_fn = SchafkopfEnv
        env_kwargs = {}
        seed = 1
        train_single_agent_models(env_fn, steps=args.steps, seed=seed, env_kwargs=env_kwargs)
    elif args.mode == "eval":
        results = evaluate_models(SchafkopfEnv, num_episodes=args.episodes)
        plot_combined_rewards(results)
        print("Evaluationsergebnisse:", results)
    elif args.mode == "console":
        main_console(episodes=args.steps, num_agents=args.agents)

if __name__ == "__main__":
    main()
