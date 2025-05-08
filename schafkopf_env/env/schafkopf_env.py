import os
import random
import time
from collections import OrderedDict
from collections import deque

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict
import numpy as np
import torch as th
import torch.nn as nn
from pettingzoo import AECEnv

from env.karte import Karte
from env.spieler import Spieler
from env.spiellogik import SpielLogik


class SchafkopfEnv(AECEnv):
    """
    Schafkopf-Umgebung für das Spiel.

    Diese Umgebung implementiert ein Schafkopf-Spiel als Multi-Agent-Environment.
    Es werden vier Spieler erstellt, ein Kartendeck generiert, Karten verteilt und
    Spielzüge gemäß den Schafkopf-Regeln verarbeitet.

    Attributes:
        spieler (list): Liste der Spielerobjekte.
        render_mode (str): Rendermodus, z. B. "human".
        blatt (list): Kartendeck, welches mit blattErstellen() erzeugt wird.
        gespielteKarten (list): Liste der bereits gespielten Karten (numerische Werte).
        state (bool): Zustand des Environments (False bis reset() aufgerufen wird).
        spielerAmZug (Spieler): Referenz auf den aktuell aktiven Spieler.
        phase (str): Aktuelle Phase ("spielartWahl" oder "kartenSpielen").
        spielart (str): Aktuelle Spielart; initial "Standart".
        spielarten (set): Menge der möglichen Spielarten (z. B. {"Rufspiel", "Passen"}).
        angesagt (str): Angesagte Spielart (falls überboten).
        spielLogik (SpielLogik): Instanz der Spiellogik, initialisiert mit der aktuellen Spielart.
        gerufene (Karte): Das im Rufspiel gerufene Ass.
        partnerAufgedeckt (bool): Flag, ob die Partneraufdeckung erfolgt ist.
        stich (dict): Enthält den aktuellen Stich.
        global_action_space (gym.Space): Global definierter Aktionsraum (Discrete(48)).
        global_observation_space (gym.Space): Global definierter Beobachtungsraum.
    """
    metadata = {"name": "schafkopf_env.py", "is_parallelizable": True, "render_modes": []}
    start_index = -1

    # ──────────────── INITIALISIERUNG ────────────────
    def __init__(self):
        """
        Initialisiert die Schafkopf-Umgebung.

        Erzeugt die Spieler, initialisiert das Kartendeck, legt Zustandsvariablen
        fest und definiert die globalen Aktions- sowie Beobachtungsräume.
        """
        self.spieler = [Spieler(position=i, rolle="unbekannt") for i in range(4)]
        self.render_mode = "human"
        self.blatt = []  # Wird später mit blattErstellen() befüllt.
        self.gespielteKarten = []  # Hier werden numerische Werte der gespielten Karten gespeichert.
        self.state = False
        self.spielerAmZug = self.spieler[0]
        self.phase = None
        self.spielart = "Standart"
        self.spielarten = {"Rufspiel", "Passen"}
        self.angesagt = None
        self.spielLogik = SpielLogik(spielArt=self.spielart)
        self.gerufene = None
        self.partnerAufgedeckt = False
        self.spielartWahlRewardVergeben = False
        self.stich = {}
        self.angespielte = None
        self.gewinnerTeam = []

        # Global definierter Aktionsraum: Discrete(48)
        self.global_action_space = gym.spaces.Discrete(48)
        self.action_spaces = {spieler.position: self.global_action_space for spieler in self.spieler}

        self.global_observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-1, high=255, shape=(33,), dtype=np.float32),
            "action_mask": gym.spaces.MultiBinary(48)
        })
        self.observation_spaces = {spieler.position: self.global_observation_space for spieler in self.spieler}

    # ──────────────── HILFSMETHODEN ────────────────
    def blattErstellen(self):
        """
        Erzeugt ein Kartendeck für das Spiel.

        Args:
            None

        Returns:
            list: Liste von Karte-Objekten, die alle Karten des Decks repräsentieren.
        """
        farben = ["Eichel", "Gras", "Herz", "Schellen"]
        symbole = ["7", "8", "9", "10", "Unter", "Ober", "König", "Ass"]
        werte = {"7": 0, "8": 0, "9": 0, "10": 10, "Unter": 2, "Ober": 3, "König": 4, "Ass": 11}
        return [Karte(farbe, symbol, werte[symbol]) for farbe in farben for symbol in symbole]

    def trumpfFestlegen(self, spielart):
        """
        Markiert Karten als Trumpf basierend auf der gewählten Spielart.

        Für Spielarten wie "Rufspiel" oder "Standart" werden alle Ober, Unter und Herz-Karten
        als Trumpf markiert.

        Args:
            spielart (str): Gewählte Spielart (z. B. "Rufspiel").

        Returns:
            None
        """
        try:
            if spielart in {"Rufspiel", "Standart"}:
                for spieler in self.spieler:
                    for karte in spieler.hand:
                        karte.istTrumpf = karte.symbol in {"Ober", "Unter"} or karte.farbe == "Herz"
            else:
                raise ValueError(f"Unbekannte Spielart: {spielart}")
        except Exception as e:
            print(f"Fehler beim Festlegen der Trumpffarbe: {e}")

    def setzePartner(self):
        """
        Weist intern die Partner zu, basierend auf dem gerufenen Ass.

        Nur der Spieler, der das gerufene Ass in der Hand hat, kennt seinen Partner.
        Geht davon aus, dass die Spielartwahl reibungslos funktioniert.

        Args:
            None

        Returns:
            None
        """
        if self.spielart == "Rufspiel":
            spielmacher = next((s for s in self.spieler if s.rolle == "Spielmacher"), None)
            sau = next((spieler for spieler in self.spieler 
                        if spieler != spielmacher and self.gerufene in spieler.hand), None)
            andere = [s for s in self.spieler if s not in (spielmacher, sau)]
            andere[0].partner = [andere[1]]
            andere[1].partner = [andere[0]]
            spielmacher.partner = [sau]
            sau.partner = [spielmacher]
            sau.rolle = self.gerufene
            andere[0].rolle = "Nicht-Spieler"
            andere[1].rolle = "Nicht-Spieler"
            self.partnerAufgedeckt = False

    def convert_stich_indices_to_cards(self, stichIdxs):
        """
        Konvertiert Stich-Einträge, die als Indizes der Handkarten gespeichert sind,
        in die entsprechenden Kartenobjekte.

        Args:
            stichIdxs (dict oder list): Dictionary oder Liste von 3-Tupeln, die die Stichinformationen enthalten.

        Returns:
            list: Liste der entsprechenden Kartenobjekte.
        """
        stich = []
        if stichIdxs is not None:
            if isinstance(stichIdxs, dict):
                for spielerPosition, value in stichIdxs.items():
                    if isinstance(value, tuple):
                        try:
                            _, _, karten_index = value
                        except Exception as e:
                            print(f"Fehler beim Entpacken des Tupels {value}: {e}")
                            continue
                        spieler = next(s for s in self.spieler if s.position == spielerPosition)
                        karte = spieler.hand[karten_index]
                        stich.append(karte)
                    else:
                        stich.append(value)
            else:
                for spielerPosition, _, karten_index in stichIdxs:
                    spieler = next(s for s in self.spieler if s.position == spielerPosition)
                    karte = spieler.hand[karten_index]
                    stich.append(karte)
        return stich

    # ──────────────── RESET UND INITIAL OBSERVATION ────────────────
    def reset(self, seed=None, options=None):
        """
        Setzt das Spiel zurück, mischt das Deck und teilt 8 Karten an jeden Spieler aus.

        Args:
            seed (optional): Zufalls-Seed für Reproduzierbarkeit.
            options (optional): Weitere Optionen für den Reset.

        Returns:
            tuple: (observations, infos) – Beobachtungen für jeden Spieler und ein leeres Infos-Dictionary.
        """
        SchafkopfEnv.start_index = (SchafkopfEnv.start_index + 1) % 4
        self.spieler = [Spieler(position=i, rolle="unbekannt") for i in range(4)]
        spieler_deque = deque(self.spieler)
        spieler_deque.rotate(-SchafkopfEnv.start_index)
        self.spieler = list(spieler_deque)
        self.blatt = self.blattErstellen()
        random.shuffle(self.blatt)
        for spieler in self.spieler:
            spieler.kartenErhalten(self.blatt.pop() for _ in range(8))
        self.trumpfFestlegen(self.spielart)
        for spieler in self.spieler:
            spieler.kartenSortieren(self.spielart)
        self.state = True
        self.gespielteKarten = []
        self.stich = {}
        self.gerufene = None
        self.partnerAufgedeckt = False
        self.spielerAmZug = self.spieler[0]
        self.phase = "spielartWahl"
        self.angesagt = None
        self.gewinnerTeam = []
        self.angespielte = None
        self.spielart = "Standart"
        self.spielartWahlRewardVergeben = False
        observations = {spieler.position: self.observe(spieler) for spieler in self.spieler}
        infos = {}
        self._cumulative_rewards = {spieler.position: 0 for spieler in self.spieler}
        return observations, infos

    # ──────────────── AKTIONSVERARBEITUNG ────────────────
    def process_spielart_action(self, spieler, action):
        """
        Verarbeitet den Aktions-Tuple zur Spielartwahl des Spielers und aktualisiert den internen Zustand.

        Args:
            spieler (Spieler): Der ausführende Spieler.
            action (tuple oder list oder np.ndarray): Aktions-Tuple in der Form (spielart, sau_index, _).

        Returns:
            tuple: Standardisierter Aktions-Tuple, z. B. ("Passen", 0, 0) für Passen oder ("Rufspiel", gerufene, 0) für Rufspiel.
        """
        if not isinstance(action, (tuple, list)):
            if isinstance(action, np.ndarray):
                if action.ndim == 1 and action.size == 3:
                    action = tuple(action.tolist())
                else:
                    flat = np.array(action).flatten()
                    action = tuple(flat[:3].tolist())
            else:
                action = (action,)
        spielart_auswahl, sau_index, _ = action

        if spielart_auswahl == "Passen" or spielart_auswahl == 0:
            return ("Passen", 0, 0)
        elif spielart_auswahl == "Rufspiel" or spielart_auswahl == 1:
            farbReihenfolge = {"Eichel": 0, "Gras": 1, "Schellen": 2}
            verfügbare_asse = sorted(
                [karte for s in self.spieler for karte in s.hand
                 if karte.symbol == "Ass" and karte.farbe != "Herz"],
                key=lambda karte: farbReihenfolge.get(karte.farbe, 99)
            )
            if 0 <= sau_index < len(verfügbare_asse):
                gerufene = verfügbare_asse[sau_index]
                self.gerufene = gerufene
                self.spielart = "Rufspiel"
                spieler.rolle = "Spielmacher"
                self.spielLogik = SpielLogik(spielArt=self.spielart)
                return ("Rufspiel", gerufene, 0)
            else:
                raise ValueError("Ungültiger Sau-Index.")

    def process_karten_action(self, spieler, action):
        """
        Verarbeitet den Aktions-Tuple zur Kartenauswahl und spielt die entsprechende Karte aus.

        Args:
            spieler (Spieler): Der ausführende Spieler.
            action (tuple oder list oder np.ndarray): Aktions-Tuple in der Form (_, _, karten_index).

        Returns:
            Karte: Die vom Spieler gespielte Karte.
        """
        if not isinstance(action, (tuple, list)):
            if isinstance(action, np.ndarray):
                if action.ndim == 1 and action.size == 3:
                    action = tuple(action.tolist())
                else:
                    flat = np.array(action).flatten()
                    action = tuple(flat[:3].tolist())
            else:
                action = (action,)
        _, _, kartenIdx = action
        if kartenIdx < 0 or kartenIdx >= len(spieler.hand):
            print(f"Warnung: Ungültiger Kartenindex {kartenIdx} für Spieler {spieler.position} (Handlänge: {len(spieler.hand)}).")
            kartenIdx = np.random.choice(range(len(spieler.hand))) if spieler.hand else 0
        karte = spieler.hand[kartenIdx]
        self.gespielteKarten.append(karte)
        return spieler.karteSpielen(karte)

    def step(self, actions):
        """
        Führt einen Trick aus, bestimmt den Gewinner und aktualisiert den Spielstatus.

        Args:
            actions (dict): Dictionary, das jedem Spieler einen Aktions-Tuple zuordnet.
                Für die Phase "spielartWahl": Tupel (spielart, sau_index, _).
                Für die Phase "kartenSpielen": Tupel zur Kartenauswahl.

        Returns:
            tuple: (observations, rewards, terminated, truncated, info)
                - observations (dict): Beobachtungen für jeden Spieler.
                - rewards (dict): Rewards für jeden Spieler.
                - terminated (dict): Kennzeichnung, ob das Spiel beendet ist.
                - truncated (dict): Kennzeichnung, ob das Spiel abgebrochen wurde.
                - info (dict): Zusätzliche Informationen.
        """
        # Round-based reset
        self.stich.clear()
        self.angespielte = None

        if not isinstance(actions, dict):
            if isinstance(actions, int):
                actions = {spieler.position: self.flat_index_to_action_tuple(actions) for spieler in self.spieler}
            else:
                actions = {spieler.position: actions for spieler in self.spieler}
        else:
            actions = {k: self.flat_index_to_action_tuple(v) if isinstance(v, int) else v for k, v in actions.items()}

        if self.phase == "spielartWahl":
            for spieler in self.spieler:
                action = actions[spieler.position]
                self.process_spielart_action(spieler, action)
            self.setzePartner()
            if self.done():
                terminated = {spieler.position: True for spieler in self.spieler}
            else:
                self.phase = "kartenSpielen"
                terminated = {spieler.position: False for spieler in self.spieler}
            truncated = {spieler.position: False for spieler in self.spieler}
            info = {}
            rewards = self.reward()
            observations = {spieler.position: self.observe(spieler) for spieler in self.spieler}
            return observations, rewards, terminated, truncated, info

        elif self.phase == "kartenSpielen":
            for spieler in self.spieler:
                if spieler.position not in actions:
                    raise KeyError(f"Kein Eintrag für Spieler {spieler.position} in actions!")
                action = actions[spieler.position]
                karte = self.process_karten_action(spieler, action)
                self.stich[spieler.position] = karte
                if self.angespielte is None:
                    self.angespielte = karte
                
            print("\n🔷 Aktueller Stich: 🔷")
            for spieler in self.spieler:
                pos = spieler.position
                if pos in self.stich:
                    print(f"Spieler {pos}: {self.stich[pos]}")
                else:
                    print(f"Spieler {pos}: Keine Karte gespielt")

            if any(karte == self.gerufene for karte in self.stich.values()):
                self.partnerAufgedeckt = True
                for spieler in self.spieler:
                    partner_str = ", ".join(f"Spieler {partner.position}" for partner in spieler.partner)
                    print(f"Partner von Spieler {spieler.position}:{partner_str}") 
              
            stichKarte = self.spielLogik.siegerErmitteln(self.stich)
            stichMacherPosition = next((spieler for spieler, karte in self.stich.items() if karte == stichKarte), None)
            stichMacher = next((s for s in self.spieler if s.position == stichMacherPosition))
            stichMacher.bekommtStichPunkte(self.stich)
            print(f"\n🏆 Spieler {stichMacherPosition} gewinnt den Stich! 🏆\n")
            self.spielerAmZug = stichMacher
            self.spieler.sort(key=lambda s: (s.position - stichMacherPosition) % 4)
            if self.done():
                terminated = {spieler.position: True for spieler in self.spieler}
            else:
                terminated = {spieler.position: False for spieler in self.spieler}
            truncated = {spieler.position: False for spieler in self.spieler}
            info = {}
            rewards = self.reward()
            observations = {spieler.position: self.observe(spieler) for spieler in self.spieler}
            return observations, rewards, terminated, truncated, info

    # ──────────────── BEOBACHTUNG & REWARD ────────────────
    def pad_or_crop(self, arr, target_length):
        """
        Passt einen eindimensionalen Array an eine Ziel-Länge an, indem er abschneidet oder mit Nullen auffüllt.

        Args:
            arr (list oder np.array): Eingabearray.
            target_length (int): Ziel-Länge des Arrays.

        Returns:
            np.array: Array mit der Länge target_length.
        """
        arr = np.array(arr, dtype=np.float32).flatten()
        if arr.shape[0] > target_length:
            return arr[:target_length]
        elif arr.shape[0] < target_length:
            return np.concatenate([arr, np.zeros(target_length - arr.shape[0], dtype=np.float32)])
        else:
            return arr

    def observe(self, spieler):
        """
        Erzeugt einen flachen Beobachtungsvektor für einen Spieler sowie die zugehörige Aktionsmaske.

        Der Vektor umfasst u. a.:
          - Hand (8 Werte)
          - Partnerstatus (1 Wert)
          - Partnerinformationen (2 Werte)
          - Numerischen Wert der gerufenen Karte (1 Wert, -1 falls nicht vorhanden)
          - Karten der aktuellen Runde (4 Werte)
          - Bereits gespielte Trümpfe (16 Werte)
          - Anzahl der Trumpfkarten in der Hand (1 Wert)

        Args:
            spieler (Spieler): Der Spieler, für den die Beobachtung erstellt wird.

        Returns:
            dict: Dictionary mit den Schlüsseln "observation" (np.array) und "action_mask" (np.array).
        """
        observation_vector = np.concatenate([
            self.pad_or_crop([karte.to_numeric() for karte in spieler.hand], 8),
            self.pad_or_crop([1 if (self.partnerAufgedeckt or self.gerufene in spieler.hand) else 0], 1),
            self.pad_or_crop([-1, -1] if not spieler.partner or not self.partnerAufgedeckt
                             else [float(p.position) for p in spieler.partner][:2], 2),
            self.pad_or_crop([self.gerufene.to_numeric() if self.gerufene is not None else -1], 1),
            self.pad_or_crop([karte.to_numeric() for karte in self.convert_stich_indices_to_cards(self.stich)], 4),
            self.pad_or_crop([karte.to_numeric() for karte in self.gespielteKarten if karte.istTrumpf], 16),
            self.pad_or_crop([len([karte for karte in spieler.hand if karte.istTrumpf])], 1)
        ])

        action_mask = np.full(48, False, dtype=bool)
        if self.phase == "spielartWahl":
            möglicheActions = self.möglicheSpielartWahl(spieler, self.stich)
        elif self.phase == "kartenSpielen":
            möglicheActions = self.möglicheKartenWahl(spieler, self.stich)
        else:
            möglicheActions = []
        if möglicheActions:
            for action_tuple in möglicheActions:
                index = self.tuple_to_index(action_tuple)
                action_mask[index] = True
        return {"observation": observation_vector, "action_mask": action_mask}

    def tuple_to_index(self, action_tuple, n1=2, n2=3, n3=8):
        """
        Konvertiert einen Aktions-Tuple (a, b, c) in einen eindeutigen Index.

        Args:
            action_tuple (tuple): Aktions-Tuple in der Form (a, b, c).
            n1 (int, optional): Anzahl erster Dimension (Default: 2).
            n2 (int, optional): Anzahl zweiter Dimension (Default: 3).
            n3 (int, optional): Anzahl dritter Dimension (Default: 8).

        Returns:
            int: Eindeutiger Index, der dem Aktions-Tuple entspricht.
        """
        a, b, c = action_tuple
        return a * (n2 * n3) + b * n3 + c

    def reward(self):
        """
        Berechnet den Reward des aktuellen Tricks und vergibt ggf. Bonus/Malus am Rundenende.

        Die Berechnung erfolgt in mehreren Schritten:
          - Sofortiger Trickreward (Stichwertung).
          - Bonus/Malus bei Spielmacher und Partner im Rufspiel.
          - Finale Belohnung basierend auf den gesammelten Punkten.

        Args:
            None

        Returns:
            dict: Reward für jeden Spieler, Schlüssel ist die Spielerposition.
        """
        if self.stich:
            self.spielartWahlRewardVergeben = True

        rewards = {spieler.position: 0 for spieler in self.spieler}

        if self.spielart == "Rufspiel" and not self.spielartWahlRewardVergeben:
            for spieler in self.spieler:
                if spieler.rolle == "Spielmacher":
                    trump_count = sum(1 for karte in spieler.hand if karte.istTrumpf)
                    if trump_count >= 5:
                        rewards[spieler.position] = 50
                    elif trump_count == 4:
                        rewards[spieler.position] = 0
                    else:
                        rewards[spieler.position] = -50
                    break

                trump_count = sum(1 for karte in spieler.hand if karte.istTrumpf)
                if trump_count >= 5:
                    rewards[spieler.position] = -50
                elif trump_count == 4:
                    rewards[spieler.position] = 0
                else:
                    rewards[spieler.position] = 50

        for spieler in self.spieler:
            if spieler.rolle == "Spielmacher":
                spielmacher = spieler

        if self.stich:
            stichKarte = self.spielLogik.siegerErmitteln(self.stich)
            stichMacherPosition = next((spieler for spieler, karte in self.stich.items() if karte == stichKarte), None)
            stichMacher = next((s for s in self.spieler if s.position == stichMacherPosition))
            stichWert = sum(karte.wert for karte in self.stich.values())
            rewards[stichMacher.position] += round(stichWert / 2)
            stichKarte = self.spielLogik.siegerErmitteln(self.stich)
            if self.angespielte == stichKarte:
                for spieler, karte in self.stich.items():
                    if karte == self.angespielte:
                        rewards[spieler] += 20
            if (stichMacher.partner is not None and self.partnerAufgedeckt) or stichMacher == spielmacher:
                for partner in stichMacher.partner:
                    if partner.position in self.stich:
                        partner_bonus = self.stich[partner.position].wert
                        rewards[partner.position] += round(partner_bonus)
            high_value_threshold = 10
            for sp, karte in self.stich.items():
                spieler = next(spieler for spieler in self.spieler if spieler.position == sp)
                if karte.wert >= high_value_threshold and karte != self.gerufene:
                    if any(self.spielLogik.istErlaubterZug(spieler, andereKarte, self.stich.values(), self.gerufene)
                           and andereKarte.wert < karte.wert for andereKarte in spieler.hand):
                        if spieler != stichMacher and (
                           (spieler not in stichMacher.partner and self.partnerAufgedeckt)
                           or (not self.partnerAufgedeckt and spieler in spielmacher.partner and stichMacher is not spielmacher)):
                            rewards[spieler.position] -= karte.wert

        if self.done():
            if self.spielart == "Rufspiel":
                spielerPartei = [s for s in self.spieler if s.rolle != "Nicht-Spieler"]
                nichtSpielerPartei = [s for s in self.spieler if s.rolle == "Nicht-Spieler"]
                punkteSP = sum(s.punkte for s in spielerPartei)
                punkteNSP = sum(s.punkte for s in nichtSpielerPartei)
                parteiBonus = 20
                if punkteSP > punkteNSP:
                    self.gewinnerTeam = spielerPartei
                    punkteVerlierer = punkteNSP
                    multiplier = 3 if punkteVerlierer == 0 else 2 if punkteVerlierer <= 30 else 1
                    for s in spielerPartei:
                        rewards[s.position] += parteiBonus * multiplier
                    for s in nichtSpielerPartei:
                        rewards[s.position] -= parteiBonus * multiplier
                else:
                    self.gewinnerTeam = nichtSpielerPartei
                    punkteVerlierer = punkteSP
                    multiplier = 3 if punkteVerlierer == 0 else 2 if punkteVerlierer < 30 else 1
                    for s in nichtSpielerPartei:
                        rewards[s.position] += parteiBonus * multiplier
                    for s in spielerPartei:
                        rewards[s.position] -= parteiBonus * multiplier

            spielmacher = next((s for s in self.spieler if s.rolle == "Spielmacher"), None)
            if spielmacher is not None:
                spielmacherBonus = 30
                if punkteSP > punkteNSP:
                    rewards[spielmacher.position] += spielmacherBonus
                else:
                    rewards[spielmacher.position] -= round(spielmacherBonus * 1.5)
            if self.spielart == "Standart":
                for spieler in self.spieler:
                    trump_count = sum(1 for karte in spieler.hand if karte.istTrumpf)
                    if trump_count >= 5:
                        rewards[spieler.position] = -20
                    elif trump_count == 4:
                        rewards[spieler.position] = 0
                    else:
                        rewards[spieler.position] = 20
        return rewards

    def done(self):
        """
        Überprüft, ob das Spiel beendet ist.

        Das Spiel gilt als beendet, wenn im Standardspiel immer True zurückgegeben wird
        oder alle Spieler keine Karten mehr besitzen.

        Args:
            None

        Returns:
            bool: True, wenn das Spiel beendet ist, sonst False.
        """
        if self.spielart == "Standart":
            return True
        if all(len(spieler.hand) == 0 for spieler in self.spieler):
            return True
        return False

    # ──────────────── SPACE-ABFRAGEN ────────────────
    def observation_space(self, agent):
        """
        Gibt den globalen Beobachtungsraum zurück.

        Args:
            agent: Der Agent (z. B. Position).

        Returns:
            gym.Space: Global definierter Beobachtungsraum.
        """
        return self.global_observation_space

    def action_space(self, agent):
        """
        Gibt den globalen Aktionsraum zurück.

        Args:
            agent: Der Agent (z. B. Position).

        Returns:
            gym.Space: Global definierter Aktionsraum.
        """
        return self.global_action_space

    # ──────────────── RENDER & CLOSE ────────────────
    def render(self):
        """
        Gibt den aktuellen Spielstand in der Konsole aus.

        Args:
            None

        Returns:
            None
        """
        print("\n")
        for spieler in self.spieler:
            print(f"{spieler}: {[str(karte) for karte in spieler.hand]}")
        print("\n")
        pass

    def close(self):
        """
        Schließt das Environment.

        Args:
            None

        Returns:
            None
        """
        pass

    # ──────────────── INTERAKTIVE TESTMETHODEN ────────────────
    def spielartWählen(self, spieler, stich):
        """
        Liest interaktiv die Spielartwahl vom Benutzer ein und gibt einen Aktions-Tuple zurück,
        ohne den internen Zustand zu ändern.

        Args:
            spieler (Spieler): Der Spieler, der die Auswahl trifft.
            stich: Aktuelle Stichinformationen (zur Ermittlung möglicher Aktionen).

        Returns:
            tuple: Aktions-Tuple, z. B. (0, 0, 0) für Passen oder (1, ass_index, 0) für Rufspiel.
        """
        print(self.möglicheSpielartWahl(spieler, stich))
        print(f"Spieler {spieler.position}, Hand: {[str(karte) for karte in spieler.hand]}")
        valid_inputs = list(self.spielarten) + ["0", "1"]
        while True:
            try:
                spielart = input(f"Möchtest du spielen? ({', '.join(valid_inputs)}): ")
                if spielart not in valid_inputs:
                    print("Ungültige Spielart! Bitte erneut wählen.")
                    continue
                elif spielart == "Passen" or spielart == "0":
                    return (0, 0, 0)
                elif (spielart == "Rufspiel" or spielart == "1") and self.angesagt == "Passen" or self.angesagt == None:
                    farbReihenfolge = {"Eichel": 0, "Gras": 1, "Schellen": 2}
                    asse = sorted(
                        [karte for s in self.spieler for karte in s.hand
                         if karte.symbol == "Ass" and karte.farbe != "Herz"],
                        key=lambda karte: farbReihenfolge.get(karte.farbe, 99)
                    )
                    print(f"Verfügbare Asse: {', '.join(str(karte) for karte in asse)}")
                    while True:
                        try:
                            assIdx = int(input(f"Wähle eine Sau (0-{len(asse)-1}): "))
                            if 0 <= assIdx < len(asse):
                                break
                            else:
                                print("Ungültiger Index! Bitte erneut wählen.")
                        except ValueError:
                            print("Bitte eine gültige Zahl eingeben.")
                    if self.spielLogik.istErlaubteSpielart(spieler, "Rufspiel", asse[assIdx]):
                        return (1, assIdx, 0)
                else:
                    print("Du musst Passen, oder das Spiel überbieten!")
                    continue
            except Exception as e:
                print(f"Ein Fehler ist aufgetreten: {e}")

    def karteWählen(self, spieler, stich):
        """
        Liest interaktiv die Kartenauswahl vom Benutzer ein und gibt einen Aktions-Tuple zurück,
        ohne den internen Zustand zu ändern.

        Args:
            spieler (Spieler): Der Spieler, der die Auswahl trifft.
            stich: Aktuelle Stichinformationen (zur Ermittlung möglicher Aktionen).

        Returns:
            tuple: Aktions-Tuple in der Form (0, 0, karten_index), wobei karten_index die gewählte Karte repräsentiert.
        """
        print(self.möglicheKartenWahl(spieler, stich))
        try:
            stich = self.convert_stich_indices_to_cards(stich)
        except Exception as e:
            print(f"Fehler beim Umwandeln der Stichkarten KW: {e}")
        print(f"Spieler {spieler.position}, Hand: {[str(karte) for karte in spieler.hand]}")
        #print(f"Bereits gespielt: {[str(karte) for karte in stich]}")
        while True:
            try:
                kartenIdx = int(input(f"Wähle eine Karte (0-{len(spieler.hand)-1}): "))
                if not (0 <= kartenIdx < len(spieler.hand)):
                    print("Ungültige Eingabe! Bitte eine Zahl im gültigen Bereich wählen.")
                    continue
                if self.spielLogik.istErlaubterZug(spieler, spieler.hand[kartenIdx], stich, self.gerufene):
                    return (0, 0, kartenIdx)
                else:
                    print("Du musst Farbe bedienen, wenn du kannst!")
                    continue
            except ValueError:
                print("Bitte eine gültige Zahl eingeben.")
            except Exception as e:
                print(f"Ein Fehler ist aufgetreten: {e}")

    def möglicheSpielartWahl(self, spieler, actions):
        """
        Gibt eine Liste möglicher Aktions-Tupel für die Spielartwahl des Spielers zurück.

        Args:
            spieler (Spieler): Der Spieler, für den die möglichen Aktionen ermittelt werden.
            actions: Aktuelle Aktionen oder Stichinformationen zur Filterung.

        Returns:
            list: Liste möglicher Aktions-Tupel.
        """
        verfügbareActions = [(0, 0, 0)]
        if actions is not None:
            for key, value in actions.items():
                if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 3:
                    if value[0] == 1:
                        return verfügbareActions
        if self.angesagt != "Rufspiel":
            farbReihenfolge = {"Eichel": 0, "Gras": 1, "Schellen": 2}
            asse = sorted(
                [karte for s in self.spieler for karte in s.hand
                 if karte.symbol == "Ass" and karte.farbe != "Herz"],
                key=lambda k: farbReihenfolge.get(k.farbe, 99)
            )
            for idx in range(min(len(asse), 3)):
                if self.spielLogik.istErlaubteSpielart(spieler, "Rufspiel", asse[idx]):
                    verfügbareActions.append((1, idx, 0))
        return verfügbareActions

    def möglicheKartenWahl(self, spieler, stich):
        """
        Gibt eine Liste möglicher Aktions-Tupel zur Kartenauswahl für den Spieler zurück.

        Args:
            spieler (Spieler): Der Spieler, für den die Kartenwahl bestimmt wird.
            stich: Aktuelle Stichinformationen, die evtl. die Wahl einschränken.

        Returns:
            list: Liste möglicher Aktions-Tupel.
        """
        try:
            stich = self.convert_stich_indices_to_cards(stich)
        except Exception as e:
            print(f"Fehler beim Umwandeln der Stichkarten MKW: {e}")
        verfügbareActions = []
        for kartenIdx in range(len(spieler.hand)):
            if self.spielLogik.istErlaubterZug(spieler, spieler.hand[kartenIdx], stich, self.gerufene):
                verfügbareActions.append((0, 0, kartenIdx))
        return verfügbareActions


# ──────────────── TEST-BLOCK ────────────────
if __name__ == "__main__":
    env = SchafkopfEnv()
    env.reset()
    done = False

    # Interaktive Phase: Spielartwahl
    actions = {}
    for spieler in env.spieler:
        actions[spieler.position] = env.spielartWählen(spieler, actions)
        if "Rufspiel" in actions[spieler.position]:
            env.angesagt = "Rufspiel"
    env.step(actions)

    if env.spielart == "Standart":
        print("Niemand wollte spielen. Das Spiel wird beendet.")
    else:
        env.trumpfFestlegen(env.spielart)
        for spieler in env.spieler:
            spieler.kartenSortieren(env.spielart)
        while not done:
            env.render()
            actions = {}
            for spieler in env.spieler:
                actions[spieler.position] = env.karteWählen(spieler, actions)
            env.step(actions)
            done = env.done()
    env.render()
    print("Spiel beendet!")
    env.close()
