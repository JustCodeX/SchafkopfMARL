class SpielLogik:
    def __init__(self, spielArt: str):
        """
        Initialisiert die Spiellogik mit der gewählten Spielart.

        Parameter:
            spielArt (str): Die gewählte Spielart (z. B. "Rufspiel").
        """
        self.spielArt = spielArt

    def punkteErmitteln(self, stich):
        """
        Berechnet die Gesamtpunkte basierend auf den Karten im Stich.

        Parameter:
            stich (dict): Dictionary der bereits gespielten Karten im aktuellen Stich.

        Rückgabe:
            int: Summe der Punkte der Karten im Stich.
        """
        return sum(karte.wert for karte in stich)

    def istErlaubteSpielart(self, spieler, spielart, ass):
        """
        Überprüft, ob der Spieler das angesagte Spiel spielen kann.

        Im Rufspiel darf der Spieler das gerufene Ass nicht in der Hand haben und
        muss mindestens eine Karte in der Farbe des Asses besitzen, die nicht Trumpf ist.

        Parameter:
            spieler: Das Spielerobjekt, dessen Hand überprüft wird.
            spielart (str): Die gewählte Spielart (z. B. "Rufspiel").
            ass: Die Ass-Karte, die ggf. gerufen wurde.

        Rückgabe:
            bool: True, wenn die Spielart erlaubt ist, sonst False.
        """
        if spielart == "Rufspiel":
            # Der Spieler darf das gerufene Ass nicht besitzen.
            for karte in spieler.hand:
                if karte is ass:
                    return False
            # Der Spieler muss mindestens eine Karte in der Farbe des Asses besitzen, die nicht Trumpf ist.
            if any(karte.farbe == ass.farbe and not karte.istTrumpf for karte in spieler.hand):
                self.spielArt = spielart
                return True
            return False

    def istErlaubterZug(self, spieler, karte, stich, gerufene):
        """
        Überprüft, ob der geplante Zug eines Spielers den Spielregeln entspricht.

        Parameter:
            spieler: Das Spielerobjekt (mit Attribut 'hand').
            karte: Die Karte, die der Spieler spielen möchte.
            stich (iterable): Sammlung der bereits gespielten Karten im aktuellen Stich.
            gerufene: Die im Rufspiel ggf. gesuchte Karte.

        Rückgabe:
            bool: True, wenn der Zug erlaubt ist, sonst False.
        """
        # Der erste Spieler im Stich darf jede Karte spielen.
        if not stich:
            return True
        
        # Bestimme die angespielte Karte (erste Karte im Stich)
        angespielte = next(karte for karte in stich)
        #print("istErlaubterZug(angespielte):", angespielte)
        handkarten = spieler.hand

        # Prüfe, ob der Spieler eine Karte in der angespielten Farbe (ohne Trumpfkarten) besitzt.
        hatAngespielteFarbe = any(
            k.farbe == angespielte.farbe and not angespielte.istTrumpf and not k.istTrumpf
            for k in handkarten
        )

        # Prüfe, ob der Spieler mindestens einen Trumpf besitzt.
        hatTrumpf = any(k.istTrumpf for k in handkarten)

        # Prüfe, ob der Spieler die gerufene Karte besitzt.
        hatGerufene = gerufene in handkarten

        if self.spielArt == "Rufspiel":
            # Wenn die angespielte Farbe der Farbe der gerufenen Karte entspricht und nicht Trumpf ist:
            if angespielte.farbe == gerufene.farbe and not angespielte.istTrumpf:
                # Der Spieler muss die gerufene Karte spielen, falls vorhanden.
                if hatGerufene and karte is not gerufene:
                    return False
                elif hatGerufene and karte is gerufene:
                    return True
            # Ist die angespielte Farbe nicht die gerufene, darf der Spieler die gerufene Karte
            # nicht spielen, sofern er mehr als eine Karte in der Hand hat.
            if angespielte.farbe != gerufene.farbe:
                if karte is gerufene and len(handkarten) > 1:
                    return False

        # Falls die angespielte Karte ein Trumpf ist und der Spieler einen Trumpf besitzt,
        # muss auch ein Trumpf gespielt werden.
        if angespielte.istTrumpf and hatTrumpf:
            return karte.istTrumpf

        # Falls der Spieler eine Karte in der angespielten Farbe besitzt, darf er nicht ausweichen.
        if hatAngespielteFarbe:
            if karte.istTrumpf:
                return False
            return karte.farbe == angespielte.farbe

        # Andernfalls ist der Zug erlaubt, wenn der Spieler keine Karte in der angespielten Farbe besitzt.
        return not any(k.farbe is angespielte for k in spieler.hand)

    def siegerErmitteln(self, stich):
        """
        Bestimmt die gewinnende Karte eines Stiches.

        Parameter:
            stich (dict): Dictionary mit Spielerpositionen als Schlüsseln und gespielten Karten als Werten.

        Rückgabe:
            Karte: Die Karte, die den Stich gewinnt.
        """
        angespielte = next(karte for karte in stich)
        # Bei Gleichstand im Kartenwert
        symbol_order = {"9": 3, "8": 2, "7": 1}
        # Sammle alle Trumpfkarten im Stich.
        trumpfKarten = [karte for karte in stich.values() if karte.istTrumpf]

        # Falls Trumpfkarten vorhanden sind und es sich um ein Rufspiel handelt, wende spezielle Logik an.
        if trumpfKarten:
            if self.spielArt == "Rufspiel":
                return self.siegerErmittelnRufspiel(trumpfKarten)

        # Ohne Trumpfkarten gewinnt die höchste Karte in der angespielten Farbe.
        angespielte = list(stich.values())[0]
        angespielteKarten = [karte for karte in stich.values() if karte.farbe == angespielte.farbe]
        return max(angespielteKarten, key=lambda karte: (karte.wert, symbol_order.get(karte.symbol, 0)))

    def siegerErmittelnRufspiel(self, trumpfKarten):
        """
        Bestimmt den Gewinner eines Rufspiels basierend auf der Trumpfreihenfolge.

        Parameter:
            trumpfKarten (list): Liste der Trumpfkarten im Stich.

        Rückgabe:
            Karte: Die Trumpfkarte, die den Stich gewinnt.
        """
        trumpfReihenfolgeRufspiel = [
            "Eichel Ober", "Gras Ober", "Herz Ober", "Schellen Ober",
            "Eichel Unter", "Gras Unter", "Herz Unter", "Schellen Unter",
            "Herz Ass", "Herz 10", "Herz König", "Herz 9", "Herz 8", "Herz 7"
        ]

        def trumpfstärkeBestimmen(karte):
            """
            Bestimmt den Rang einer Karte basierend auf der Trumpfreihenfolge.

            Rückgabe:
                int: Index in der Trumpfreihenfolge.
            """
            return trumpfReihenfolgeRufspiel.index(str(karte))

        # Die Karte mit dem niedrigsten Index (höchster Rang) gewinnt.
        return min(trumpfKarten, key=trumpfstärkeBestimmen)
