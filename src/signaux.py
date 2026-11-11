"""
signaux.py - Génération de signaux de test (notes synthétiques) et lecture WAV.

Sert à :
  * tester l'algorithme sans micro (tests unitaires, mode --simulation) ;
  * produire les figures du rapport.
"""
import time
import wave

import numpy as np

FS_DEFAUT = 44100


def note_synthetique(f0, duree=0.5, fs=FS_DEFAUT, harmoniques=(1.0, 0.6, 0.4, 0.25, 0.12),
                     bruit=0.01, rng=None):
    """Son de type instrument : fondamentale + harmoniques + un peu de bruit."""
    rng = np.random.default_rng(0) if rng is None else rng
    t = np.arange(int(duree * fs)) / fs
    x = np.zeros_like(t)
    for h, amp in enumerate(harmoniques, start=1):
        if h * f0 < fs / 2:
            x += amp * np.sin(2 * np.pi * h * f0 * t)
    x /= max(np.max(np.abs(x)), 1e-9)
    x *= 0.5
    return x + bruit * rng.standard_normal(len(t))


def lire_wav(chemin):
    """Lit un WAV PCM 16 bits (mono ou stéréo -> mono). Retourne (signal float [-1, 1], fs)."""
    with wave.open(chemin, "rb") as w:
        fs, canaux, largeur = w.getframerate(), w.getnchannels(), w.getsampwidth()
        brut = w.readframes(w.getnframes())
    if largeur != 2:
        raise ValueError("Seuls les WAV PCM 16 bits sont gérés.")
    x = np.frombuffer(brut, dtype="<i2").astype(float) / 32768.0
    if canaux > 1:
        x = x.reshape(-1, canaux).mean(axis=1)
    return x, fs


class SourceSimulee:
    """Fait semblant d'être un micro : joue une suite de notes légèrement désaccordées.

    Même interface que SourceMicro : lire(n) renvoie les n derniers échantillons.
    """
    SEQUENCE = [  # (fréquence de la note juste, désaccord en cents)
        (82.41, 0), (110.00, -18), (146.83, +12), (196.00, -6),
        (246.94, +25), (329.63, 0),
    ]
    DUREE_NOTE = 4.0  # secondes par note

    def __init__(self, fs=FS_DEFAUT, horloge=time.time):
        self.fs = fs
        self._horloge = horloge
        self._t0 = horloge()
        self._rng = np.random.default_rng(1)

    def demarrer(self):
        self._t0 = self._horloge()

    def arreter(self):
        pass

    def lire(self, n):
        t_fin = self._horloge() - self._t0
        t = t_fin - (n - 1 - np.arange(n)) / self.fs
        idx = np.floor(t / self.DUREE_NOTE).astype(int)
        x = np.zeros(n)
        for i in np.unique(idx):
            if i < 0:
                continue
            f_juste, cents = self.SEQUENCE[i % len(self.SEQUENCE)]
            f = f_juste * 2 ** (cents / 1200)
            m = idx == i
            tl = t[m] - i * self.DUREE_NOTE
            s = sum(a * np.sin(2 * np.pi * (h * f) * tl)
                    for h, a in enumerate((1.0, 0.6, 0.4, 0.25), start=1))
            x[m] = 0.2 * s * np.minimum(1.0, tl / 0.05)  # petite attaque
        return x + 0.003 * self._rng.standard_normal(n)
