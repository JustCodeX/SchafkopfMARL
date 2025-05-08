"SchafkopfEnv" 

# def flatten_observation(self, obs):
#     """
#     Erwartet, dass obs ein Dictionary mit den folgenden Schlüsseln ist:
#     - "hand": Liste von Kartenobjekten (wird in 8 numerische Werte konvertiert)
#     - "partnerStatus": 1 Zahl
#     - "partner": 1 Zahl (oder passe target_length an, wenn du mehr Zahlen möchtest)
#     - "stichGewinner": 1 Zahl
#     - "gespielteKarten": Liste von Kartenobjekten (wird in numerische Werte konvertiert; feste Länge 7)
#     - "action_mask": Liste (48 Werte)

#     Hier werden den Karten außerdem numerische Werte zugewiesen
#     """
#     # Hand: 8 Werte
#     hand = self.pad_or_crop([karte.to_numeric() for karte in obs["hand"]], 8)
#     # partnerStatus: 1 Wert
#     partnerStatus = self.pad_or_crop([obs["partnerStatus"]], 1)
#     # partner: Jetzt 2 Werte (kein zusätzliches Nesting, da obs["partner"] ist bereits eine Liste)
#     partner = self.pad_or_crop(obs["partner"], 2)
#     # stichGewinner: 1 Wert
#     stichGewinner = self.pad_or_crop([obs["stichGewinner"]], 1)
#     # gespielteKarten: z. B. 8 Werte
#     gespielteKarten = self.pad_or_crop([karte.to_numeric() for karte in obs["gespielteKarten"]], 8)
#     # action_mask: 48 Werte
#     action_mask = self.pad_or_crop(obs["action_mask"], 48)
    
#     # Gesamt: 8 + 1 + 2 + 1 + 8 + 48 = 68
#     return np.concatenate([hand, partnerStatus, partner, stichGewinner, gespielteKarten, action_mask])

# def observe(self, spieler):
#     """
#     Erzeugt einen flachen Beobachtungsvektor für einen Spieler, inkl. Aktionsmaske.

#     Abhängig von der Phase werden unterschiedliche zulässige Aktionen ermittelt.
#     Die einzelnen Zustandskomponenten (Hand, Partnerstatus, Stichgewinner,
#     gespielte Karten und Aktionsmaske) werden dann in einen flachen Vektor überführt.

#     Returns:
#         np.array: Flacher Beobachtungsvektor (Shape: (66,)).
#     """
#     if self.phase == "spielartWahl":
#         möglicheActions = self.möglicheSpielartWahl(spieler)
#     elif self.phase == "kartenSpielen":
#         möglicheActions = self.möglicheKartenWahl(spieler, self.stich)
#     else:
#         möglicheActions = []

#     gesamtAktionen = 2 * 3 * 8  # 48 mögliche Aktionen

#     def actionZuIndex(actionsTupel):
#         a, b, c = actionsTupel
#         if isinstance(a, str):
#             a = 0 if a == "Passen" else 1
#         return a * (3 * 8) + b * 8 + c

#     mask = [0] * gesamtAktionen
#     for action in möglicheActions:
#         idx = actionZuIndex(action)
#         if idx < gesamtAktionen:
#             mask[idx] = 1
#         else:
#             print(f"Warnung: Aktionsindex {idx} außerhalb des Bereichs (max {gesamtAktionen-1}).")

#     partnerStatus = 1 if (self.partnerAufgedeckt or self.gerufene in spieler.hand) else 0
#     stichGewinner = self.spielerAmZug.position if self.spielerAmZug is not None else -1
#     # Partner: Falls keine Partner vorhanden, verwende Platzhalter [-1, -1]
#     partner = [-1, -1]
#     for i, p in enumerate(spieler.partner):
#         if i < 2:
#             partner[i] = float(p.position)

#     # Erstelle das Beobachtungs-Dictionary und wandle es in einen flachen Vektor um
#     obs_dict = {
#         "hand": spieler.hand,                
#         "partnerStatus": partnerStatus,
#         "partner": partner,
#         "stichGewinner": stichGewinner,
#         "gespielteKarten": self.gespielteKarten,  
#         "action_mask": mask
#     }
#     return self.flatten_observation(obs_dict)

"PPOEnv"

# def train_separate_policies(env_fn, total_timesteps=2048, seed=0, num_episodes=100, **env_kwargs):
#     # Zunächst erzeugen wir eine temporäre Instanz, um die Agent-IDs zu ermitteln.
#     temp_env = env_fn(**env_kwargs)
#     agent_ids = temp_env.possible_agents  # z. B. [0,1,2,3]
#     temp_env.close()

#     models = {}
#     single_envs = {}
#     policy_kwargs = {}  # Hier kannst du weitere Parameter für die Policy setzen

#     # Für jeden Agenten: Erzeuge ein neues Multiagenten-Environment, wickle es in SB3ActionMaskWrapper und dann in SingleAgentWrapper.
#     for agent in agent_ids:
#         def make_env(agent_id=agent):
#             env = env_fn(**env_kwargs)  # Neue Instanz des Multiagenten-Environments
#             # Wickle in deinen SB3ActionMaskWrapper ein (sofern implementiert)
#             env = SB3ActionMaskWrapper(env)
#             return SingleAgentWrapper(env, agent_id)
#         # Erzeuge ein DummyVecEnv – übergebe eine Funktion, die immer eine neue Instanz erzeugt.
#         single_env = DummyVecEnv([make_env])
#         single_envs[agent] = single_env

#         models[agent] = MaskablePPO(
#             MaskableActorCriticPolicy,
#             single_env,
#             policy_kwargs=policy_kwargs,
#             verbose=1,
#             seed=seed + int(agent)
#         )
    
#     # Training über mehrere Episoden
#     for episode in range(num_episodes):
#         for agent in agent_ids:
#             obs, _ = single_envs[agent].reset()  # Kein seed-Argument hier, da DummyVecEnv das nicht unterstützt
#             done = False
#             while not done:
#                 action, _states = models[agent].predict(obs, deterministic=True)
#                 obs, reward, done, info = single_envs[agent].step(action)
#             # Nach jeder Episode kannst du optional ein learn() aufrufen – je nach Trainingsstrategie:
#             models[agent].learn(total_timesteps=total_timesteps)
#         print(f"Episode {episode} abgeschlossen.")
    
#     # Speichere alle Modelle in einem Ordner
#     save_folder = "models"
#     os.makedirs(save_folder, exist_ok=True)
#     for agent in agent_ids:
#         save_path = os.path.join(save_folder, f"agent_{agent}_{time.strftime('%Y%m%d-%H%M%S')}")
#         models[agent].save(save_path)
#         print(f"Modell für Agent {agent} wurde gespeichert unter {save_path}.")
    
#     # Schließe alle Umgebungen
#     for agent in agent_ids:
#         single_envs[agent].close()
    
#     return models

# class SingleAgentWrapper(gym.Env):
#     def __init__(self, env, agent_id):
#         """
#         env: Das Multiagenten-Environment (z. B. bereits mit SB3ActionMaskWrapper gewrappt)
#         agent_id: Die ID des Agenten (z. B. 0, 1, 2, 3)
#         """
#         super(SingleAgentWrapper, self).__init__()
#         self.env = env
#         self.agent_id = agent_id
#         # Wir gehen davon aus, dass der SB3ActionMaskWrapper einen Dict-Space liefert,
#         # der z. B. so aussieht:
#         #   gym.spaces.Dict({
#         #       "observation": Box(...), "action_mask": Box(...)
#         #   })
#         # Da alle Agenten in der Multiagentenv denselben Raum haben, übernehmen wir ihn einfach.
#         self.observation_space = env.observation_space  
#         self.action_space = env.action_space  

#     def reset(self, seed=None, options=None):
#         observations, info = self.env.reset(seed=seed, options=options)
#         # Setze den aktiven Agenten auf den gewünschten Agenten.
#         self.env.agent_selection = self.agent_id
#         # Hole die Beobachtung über den Wrapper, der ein Dictionary mit "observation" und "action_mask" liefert.
#         obs = self.env._get_obs(self.agent_id)
#         return obs, info

#     def step(self, action):
#         # Erstelle ein Aktions-Dictionary für alle Agenten.
#         actions = {}
#         for a in self.env.possible_agents:
#             # Für unseren Agenten verwende die übergebene Aktion,
#             # für alle anderen setzen wir einen Dummy-Wert (hier 0 – passe das je nach Umgebung an).
#             actions[a] = action if a == self.agent_id else 0
#         observations, rewards, dones, infos = self.env.step(actions)
#         # Erzwinge, dass die Beobachtung des aktiven Agenten zurückgegeben wird:
#         obs = self.env._get_obs(self.agent_id)
#         return obs, rewards[self.agent_id], dones[self.agent_id], infos

#     def render(self):
#         return self.env.render()

#     def close(self):
#         self.env.close()

# class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env): 
#     def _get_obs(self, agent):
#       spieler_obj = self.get_spieler(agent)
#       obs = self.env.observe(spieler_obj)
#       observation_vector = np.array(obs["observation"], dtype=np.float32)
#       action_mask = np.array(obs["action_mask"], dtype=np.bool_)
#       return {"observation": observation_vector, "action_mask": action_mask}