# README.md

# Schafkopf PPO Implementierung

Eine Python-Implementierung von **Proximal Policy Optimization (PPO)** für das traditionelle bayerische Kartenspiel **Schafkopf**.

## Übersicht

Dieses Projekt bietet:

- Eine erweiterbare Schafkopf-Umgebung für Multi-Agent Reinforcement Learning (MARL) basierend auf PettingZoo (`schafkopf_env.py`)
- Eine Gym-kompatible Umgebung mit Implementierung des PPO-Algorithmus zum Training und zur Evaluierung von PPO-Agenten (`schafkopfPPOSingleEnv.py`)
- Skripte zum Training und zur Evaluierung:
  - Gegen trainierte Agents spielen
  - Agents selbst trainieren
  - Verschiedene Versionen von Agents gegeneinander antreten lassen
- Logging via TensorBoard für Trainings- und Evaluierungsmetriken

## Projektstruktur

```
Schafkopf_Env/                    # Projekt-Root
├── schafkopf_env/               # PettingZoo-MARL-Umgebung
│   ├── consoleResults/          # Konsolenergebnisse und Logs
│   └── env/                     # Environment-Komponenten
│       ├── karte.py             # Karten-Logik
│       ├── schafkopf_env.py     # Kern-Environment
│       ├── spieler.py           # Spieler-Interface
│       └── spiellogik.py        # Spielregeln und -ablauf
├── frontend/                    # Web-Frontend (FastAPI + Uvicorn)
├── Graphen/                     # gespiecherte Evaluations-Graphen
├── models/                      # Gespeicherte Modelle
├── tensorboard_logs/            # TensorBoard-Logs
├── schafkopf_env_v0.py          # Alte Version der Umgebung
├── schafkopfPPOenv.py           # Gym-kompatible PPO-Umgebung v1
├── schafkopfPPOSingleEnv.py     # Gym-PPO-Umgebung für Single-Agent
├── schafkopfPPOEnv2.py          # Gym-PPO-Umgebung v2
├── README.md                    # Projektbeschreibung
└── requirements.txt             # Python-Abhängigkeiten
```

## Installation

1. Repository klonen
   ```bash
   git clone https://github.com/JustCodeX/SchafkopfMARL.git
   cd schafkopf_env
   ```
2. Virtuelle Umgebung erstellen & aktivieren

   **Mit `venv`**
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

   **Mit `conda`**
   ```bash
   conda create -n schafkopf python=3.8 -y
   conda activate schafkopf
   ```
   
3. Abhängigkeiten installieren
   ```bash
   pip install -r requirements.txt
   ```

## Schnellstart

### Training

Starte das Training mit Standard-Hyperparametern:
path: Schafkopf_Env/schafkopf_env
```bash
python schafkopfPPOSingleEnv.py --mode train --steps <num_steps>
```

### Evaluation

Bewerte ein gespeichertes Modell:
path: Schafkopf_Env/schafkopf_env
```bash
python schafkopfPPOSingleEnv.py --mode eval --episodes <num_episodes>
```

Spiele gegen gespeicherte Modelle:
path: Schafkopf_Env/schafkopf_env
```bash
python schafkopfPPOSingleEnv.py --mode console --episodes <num_episodes> --agents <num_agents>
```

### Visualisierung beim Spiel gegen gespeicherte Modelle

path: Schafkopf_Env/schafkopf_env/frontend
```bash
python -m http.server 5500
```

Im Browser aufrufen:
(lokal)
```bash
http://localhost:5500 
```
(andere Geräte im Netzwerk)
```bash
http://<IP-Adresse>:5500 
```


## Konfiguration

Hyperparameter können direkt über CLI-Optionen angepasst werden:

| Option              | Beschreibung                            | Standard   |
|---------------------|-----------------------------------------|------------|
| `--episodes`        | Anzahl der Trainingsepisoden            | `10000`    |
| `--learning-rate`   | Lernrate                                | `0.0003`   |
| `--gamma`           | Diskontfaktor                           | `0.99`     |
| `--clip-epsilon`    | PPO-Clipping-Epsilon                    | `0.2`      |
| `--log-dir`         | Verzeichnis für TensorBoard-Logs        | `data/logs`|

Weitere Parameter siehe schafkopfPPOSingleEnv.py `train_single_agent_models(env_fn, steps, seed, env_kwargs={})`.

