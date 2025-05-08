class Karte:
    def __init__(self, farbe: str, symbol: str, wert: int, istTrumpf: bool = False):
        """
        Erstellt ein Karten‑Objekt.

        Parameters:
            farbe (str): Die Farbe der Karte (z. B. "Eichel", "Gras", "Herz", "Schellen").
            symbol (str): Das Symbol der Karte (z. B. "Ass", "10", "König", "Ober", "Unter", "9", "8", "7").
            wert (int): Der Punktwert der Karte.
            istTrumpf (bool): Gibt an, ob die Karte Trumpf ist.
        """
        self.farbe = farbe
        self.symbol = symbol
        self.wert = wert
        self.istTrumpf = istTrumpf

    def to_numeric(self) -> int:
        """
        Wandelt das Karten‑Objekt in einen einzelnen numerischen Wert um.

        Die Umwandlung basiert auf folgender Codierung:
            - Farbe: Eichel=0, Gras=1, Herz=2, Schellen=3 (2 Bits)
            - Symbol (Rang): Annahme: Ass=0, 10=1, König=2, Ober=3, Unter=4, 9=5, 8=6, 7=7 (3 Bits)
            - Trumpf‑Status: 0 = nicht Trumpf, 1 = Trumpf (1 Bit)

        Die finale Zahl wird folgendermaßen zusammengesetzt:
            numeric = (farbe << 4) | (symbol << 1) | trumpf_bit

        Returns:
            int: Der numerische Wert der Karte (zwischen 0 und 63).
        """
        farben = {"Eichel": 0, "Gras": 1, "Herz": 2, "Schellen": 3}
        sym_order = {"Ass": 0, "10": 1, "König": 2, "Ober": 3, "Unter": 4, "9": 5, "8": 6, "7": 7}
        color_idx = farben[self.farbe]
        symbol_idx = sym_order[self.symbol]
        trumpf_bit = 1 if self.istTrumpf else 0

        numeric = (color_idx << 4) | (symbol_idx << 1) | trumpf_bit
        return numeric

    @staticmethod
    def from_numeric(num: int) -> "Karte":
        """
        Erzeugt ein Karten‑Objekt aus einem numerischen Wert.

        Die Umwandlung erfolgt invers zu to_numeric():
            - Die untersten 1 Bit entsprechen dem Trumpf‑Status.
            - Die nächsten 3 Bits repräsentieren den Symbol‑Index.
            - Die obersten 2 Bits repräsentieren den Farbe‑Index.

        Parameters:
            num (int): Der numerische Wert der Karte (0 bis 63).

        Returns:
            Karte: Das rekonstruierte Karten‑Objekt.
        """
        trumpf_bit = num & 1
        symbol_idx = (num >> 1) & 0b111  # 3 Bits für das Symbol
        color_idx = (num >> 4) & 0b11      # 2 Bits für die Farbe

        farben = ["Eichel", "Gras", "Herz", "Schellen"]
        sym_order = ["Ass", "10", "König", "Ober", "Unter", "9", "8", "7"]
        farbe = farben[color_idx]
        symbol = sym_order[symbol_idx]

        values = {"Ass": 11, "10": 10, "König": 4, "Ober": 3, "Unter": 2, "9": 0, "8": 0, "7": 0}
        wert = values[symbol]
        istTrumpf = trumpf_bit == 1

        return Karte(farbe, symbol, wert, istTrumpf)

    def __str__(self):
        """
        Gibt die Karte als lesbare Zeichenkette zurück.

        Returns:
            str: Darstellung der Karte, z. B. "Herz Ass".
        """
        return f"{self.farbe} {self.symbol}"

    def __eq__(self, other):
        """
        Vergleicht diese Karte mit einer anderen Karte basierend auf Farbe, Symbol, Wert und Trumpfstatus.

        Parameters:
            other: Ein anderes Karten-Objekt.

        Returns:
            bool: True, wenn alle Attribute übereinstimmen, sonst False.
        """
        if isinstance(other, Karte):
            return (self.farbe == other.farbe and
                    self.symbol == other.symbol and
                    self.wert == other.wert and
                    self.istTrumpf == other.istTrumpf)
        return False
