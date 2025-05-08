import random

class Spieler:
    def __init__(self, position: float, rolle: str):
        """
        Erstellt einen Spieler mit einer Position, Rolle und initialen Werten.

        Parameter:
            position (float): Die Position des Spielers (z. B. für Partnerkopplung).
            rolle (str): Die anfängliche Rolle des Spielers (z. B. "Vorhand", "Mittelhand", "Hinterhand").

        Attributes:
            hand (list): Liste der Karten in der Hand des Spielers.
            position (float): Die Position des Spielers.
            rolle (str): Die aktuelle Rolle des Spielers.
            punkte (float): Aktuelle Punktzahl des Spielers.
            partner (list): Liste der Partner-Spieler.
            actions (list): Liste möglicher Aktionen (optional).
        """
        self.hand = []
        self.position = position
        self.rolle = rolle
        self.punkte = 0.0
        self.partner = []
        self.actions = []

    def kartenErhalten(self, karten):
        """
        Fügt dem Spieler die übergebene Sammlung an Karten zur Hand hinzu.

        Parameter:
            karten (iterable): Eine Sammlung von Kartenobjekten, die dem Spieler zugeteilt werden.
        """
        self.hand.extend(karten)

    def karteSpielen(self, karte):
        """
        Spielt die angegebene Karte, indem sie aus der Hand des Spielers entfernt wird.

        Parameter:
            karte: Das Kartenobjekt, das gespielt werden soll.

        Rückgabe:
            Das gespielte Kartenobjekt, falls es in der Hand gefunden wurde, sonst None.
        """
        if karte in self.hand:
            self.hand.remove(karte)
            # print(karte)
            # print(f"Spieler {self.position}, Hand: {[str(k) for k in self.hand]}")
            return karte
        return None

    def bekommtStichPunkte(self, stich):
        """
        Erhöht die Punktzahl des Spielers anhand der Werte der Karten im übergebenen Stich.

        Parameter:
            stich (dict): Dictionary, das die gespielten Karten (als Werte) zu den jeweiligen Spielerpositionen (als Schlüssel) enthält.
        """
        for _, karte in stich.items():
            self.punkte += karte.wert

    def kartenSortieren(self, spielart):
        """
        Sortiert die Handkarten des Spielers anhand der Trumpfstärke und Farbe.
        
        Sortierung:
        - Zuerst alle Trumpfkarten, sortiert nach ihrer Trumpfstärke (basierend auf einer vordefinierten Trumpfreihenfolge).
        - Danach alle anderen Karten, sortiert nach Farbe (Eichel > Gras > Herz > Schellen) und innerhalb einer Farbe:
            nach ihrem numerischen Wert, und falls mehrere Karten denselben Wert haben,
            nach einer vordefinierten Symbolreihenfolge (z. B. 9 > 8 > 7).
        
        Parameter:
            spielart (str): Die aktuelle Spielart, die ggf. die Sortierung (z. B. Trumpffestlegung) beeinflusst.
        """
        # Farbreihenfolge definieren (niedriger Wert = höhere Priorität)
        farb_reihenfolge = {"Eichel": 1, "Gras": 2, "Herz": 3, "Schellen": 4}
        
        # Optional: Eine zusätzliche Symbolreihenfolge (z. B. für nicht‑Trumpfkarten, um gleiche Zahlenwerte weiter zu sortieren)
        # Hier als Beispiel: Bei gleichen Zahlenwerten soll 9 vor 8 vor 7 sortiert werden. (Passen Sie die Reihenfolge ggf. an.)
        symbol_reihenfolge = {
            "Ass": 6,
            "10": 5,
            "König": 4,
            "Ober": 3,
            "Unter": 2,
            "9": 1,
            "8": 0,
            "7": -1
        }
        
        def trumpfstärkeBestimmen(karte, trumpfReihenfolge):
            """
            Bestimmt den Rang einer Trumpfkarte anhand der übergebenen Trumpfreihenfolge.

            Parameter:
                karte: Das zu bewertende Kartenobjekt.
                trumpfReihenfolge (list): Liste, die die Trumpfreihenfolge als Strings definiert.

            Rückgabe:
                int: Der Index der Karte in der Trumpfreihenfolge.
            """
            return trumpfReihenfolge.index(str(karte))

        trumpfReihenfolgeRufspiel = [
            "Eichel Ober", "Gras Ober", "Herz Ober", "Schellen Ober",
            "Eichel Unter", "Gras Unter", "Herz Unter", "Schellen Unter",
            "Herz Ass", "Herz 10", "Herz König", "Herz 9", "Herz 8", "Herz 7"
        ]

        def sortierschluesselRufspiel(karte):
            # Für Trumpfkarten: Verwende den Rang in der festgelegten Trumpfreihenfolge.
            if karte.istTrumpf:
                return (0, trumpfstärkeBestimmen(karte, trumpfReihenfolgeRufspiel))
            # Für Nicht‑Trumpfkarten: Sortiere zuerst nach Farbe, dann nach dem Zahlenwert und schließlich (bei Gleichstand) nach der Symbolreihenfolge.
            return (1, farb_reihenfolge[karte.farbe], karte.wert, symbol_reihenfolge.get(karte.symbol, 0))

        # Für die Spielarten "Rufspiel" und "Standart" wird die Sortierung angewendet.
        if spielart in {"Rufspiel", "Standart"}:
            self.hand.sort(key=sortierschluesselRufspiel)

    def __str__(self):
        """
        Gibt den Spieler als lesbare Zeichenkette zurück.

        Rückgabe:
            str: Zeichenkette, die Position, Rolle und Punktzahl des Spielers darstellt.
        """
        partner_str = ", ".join(f"Spieler {partner.position}" for partner in self.partner)
        return f"Spieler(Position: {self.position}, Rolle: {self.rolle}, Partner: {partner_str}, Punkte: {self.punkte})"

    def __eq__(self, other):
        """
        Vergleicht diesen Spieler mit einem anderen Spieler basierend auf der Position.

        Parameter:
            other: Ein anderes Spieler-Objekt.

        Rückgabe:
            bool: True, wenn 'other' ein Spieler ist und die Position übereinstimmt, sonst False.
        """
        if isinstance(other, Spieler):
            return self.position == other.position
        return False

    def __hash__(self):
        """
        Liefert einen Hash-Wert für den Spieler, basierend auf der Position.

        Rückgabe:
            int: Hash-Wert.
        """
        return hash(self.position)

###Heuristische Umsetung 
    # def heuristischeSpielartWahl(self, verfügbareActions):
    #     """
    #     Bestimmt anhand einer Heuristik die Spielartwahl des Spielers und gibt ein Aktions-Tuple zurück,
    #     basierend auf dem übergebenen Aktionsraum (verfügbareActions).

    #     Heuristik:
    #     - Der Spieler wählt "Rufspiel", wenn er mindestens 4 Trumpfkarten in seiner Hand hat.
    #     - Für "Rufspiel" wird das Ass der Farbe gewählt, in der der Spieler (unter den Nicht-Trumpfkarten)
    #         die wenigsten Karten besitzt.
    #     - Ist die Bedingung (mindestens 4 Trumpfkarten) nicht erfüllt, wird "Passen" gewählt.

    #     Parameters:
    #         verfügbareActions (list): Liste möglicher Aktions-Tupel, z.B.
    #                                 [("Passen", 0, 0), ("Rufspiel", 0, 0), ("Rufspiel", 1, 0), ("Rufspiel", 2, 0)].
        
    #     Returns:
    #         tuple: Das ausgewählte Aktions-Tuple.
    #     """
    #     # Zähle die Trumpfkarten in der Hand des Spielers.
    #     trumpfAnzahl = sum(1 for karte in self.hand if karte.istTrumpf)
    #     if trumpfAnzahl < 4:
    #         # Falls weniger als 4 Trumpfkarten vorhanden sind, wähle "Passen" (sofern vorhanden).
    #         for action in verfügbareActions:
    #             if action[0] == "Passen":
    #                 return action
    #         # Fallback: Gib das erste verfügbare Tupel zurück.
    #         return verfügbareActions[0]

    #     # Falls mindestens 4 Trumpfkarten vorhanden sind, wähle "Rufspiel".
    #     # Untersuche die Nicht-Trumpfkarten, um die Farbe zu finden, in der der Spieler am wenigsten Karten hat.
    #     nichtTrumpfFarben = ["Eichel", "Gras", "Schellen"]
    #     farbZähler = {farbe: 0 for farbe in nichtTrumpfFarben}
    #     for karte in self.hand:
    #         if not karte.istTrumpf and karte.farbe in nichtTrumpfFarben:
    #             farbZähler[karte.farbe] += 1

    #     # Wähle die Farbe mit der minimalen Anzahl; bei Gleichstand wird die Reihenfolge der Liste genutzt.
    #     geringsteFarbe = min(nichtTrumpfFarben, key=lambda s: farbZähler[s])
    #     # Bestimme den Index der gewählten Farbe in einer fest definierten Reihenfolge.
    #     farbOrdnung = {"Eichel": 0, "Gras": 1, "Schellen": 2}
    #     farbenIdx = farbOrdnung[geringsteFarbe]

    #     # Suche im Aktionsraum nach einer Aktion für "Rufspiel" mit diesem sau_index.
    #     for action in verfügbareActions:
    #         if action[0] == "Rufspiel" and action[1] == farbenIdx:
    #             return action

    #     # Falls keine Aktion mit dem gewünschten sau_index verfügbar ist, wähle die erste "Rufspiel"-Aktion.
    #     for action in verfügbareActions:
    #         if action[0] == "Rufspiel":
    #             return action

    #     # Fallback: Falls gar keine passende Aktion gefunden wird, gib das erste Tupel zurück.
    #     return verfügbareActions[0]

    # def heuristic_karten_action(self, spieler, stich):
    #     """
    #     Bestimmt anhand einer einfachen Heuristik die Kartenauswahl für einen Spieler.

    #     Parameters:
    #         spieler (Spieler): Der Spieler, der eine Karte auswählen soll.
    #         stich (dict): Dictionary der bereits gespielten Karten im aktuellen Stich.
            
    #     Returns:
    #         tuple: Ein Aktions-Tuple (0, 0, karte), wobei 'karte' das gewählte Kartenobjekt ist.
    #     """
    #     # Falls schon Karten im Stich sind, ermittele die angespielte Farbe
    #     if stich:
    #         angespielte = list(stich.values())[0]
    #         # Finde alle Karten in der Hand, die der angespielten Farbe entsprechen und regelkonform gespielt werden können
    #         passendeKarten = [karte for karte in spieler.hand
    #                         if karte.farbe == angespielte.farbe and 
    #                         self.spielLogik.istErlaubterZug(spieler, karte, stich, self.gerufene)]
    #         if passendeKarten:
    #             # Wähle die Karte mit dem geringsten Wert (oder eine andere Strategie)
    #             gewählte = min(passendeKarten, key=lambda k: k.wert)
    #             return (0, 0, gewählte)
        
    #     # Falls keine passende Farbe vorhanden ist oder kein Stich begonnen hat,
    #     # wähle die legalste (z. B. die Karte mit dem geringsten Wert, die regelkonform ist)
    #     legaleKarten = [karte for karte in spieler.hand
    #                     if self.spielLogik.istErlaubterZug(spieler, karte, stich, self.gerufene)]
    #     if legaleKarten:
    #         gewählte = min(legaleKarten, key=lambda k: k.wert)
    #         return (0, 0, gewählte)
        
    #     # Falls keine Karte regelkonform gespielt werden kann (sollte eigentlich nicht passieren)
    #     return (0, 0, spieler.hand[0])

