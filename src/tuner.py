"""
tuner.py - Accordeur chromatique en temps réel.

Utilisation (depuis la racine du dépôt) :
    python -m src.tuner                    # micro + fenêtre graphique
    python -m src.tuner --terminal         # micro + affichage dans le terminal
    python -m src.tuner --simulation       # sans micro : notes synthétiques
    python -m src.tuner --simulation --capture docs/captures/capture.png
    python -m src.tuner --liste-micros     # liste les périphériques audio

Chaîne de traitement, répétée ~15 fois par seconde :
    micro -> tampon circulaire -> dernier bloc de N échantillons
          -> fenêtre de Hann -> FFT -> spectre
          -> HPS + interpolation -> f0 -> note + écart en cents
          -> lissage (médiane des dernières mesures) -> affichage
"""
import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dsp import calculer_spectre, estimer_f0, freq_vers_note, niveau_rms
from .signaux import FS_DEFAUT, SourceSimulee

TAILLE_BLOC = 8192          # 8192 / 44100 = 0,19 s -> résolution de 5,4 Hz par bin (avant zero-padding)
SEUIL_SILENCE = 0.005       # niveau RMS en dessous duquel on considère qu'il n'y a pas de son
TOLERANCE_CENTS = 5.0       # |écart| <= 5 cents -> "accordé"


# --------------------------------------------------------------------------
# Sources audio
# --------------------------------------------------------------------------
class SourceMicro:
    """Capture le micro avec la bibliothèque sounddevice (PortAudio).

    Le micro appelle _callback() chaque fois qu'un petit paquet d'échantillons
    est prêt ; on les range dans un tampon circulaire de taille TAILLE_BLOC.
    lire() renvoie une copie de ce tampon (les derniers échantillons captés).
    """

    def __init__(self, fs=FS_DEFAUT, taille=TAILLE_BLOC, peripherique=None):
        self.fs = fs
        self._tampon = np.zeros(taille)
        self._verrou = threading.Lock()
        self._peripherique = peripherique
        self._flux = None

    def _callback(self, indata, frames, temps, statut):
        mono = indata[:, 0]
        with self._verrou:
            self._tampon = np.roll(self._tampon, -len(mono))
            self._tampon[-len(mono):] = mono

    def demarrer(self):
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            sys.exit("sounddevice / PortAudio introuvable : pip install sounddevice\n"
                     f"({e})\nAstuce : essayez --simulation pour tester sans micro.")
        self._flux = sd.InputStream(samplerate=self.fs, channels=1, dtype="float32",
                                    device=self._peripherique, callback=self._callback,
                                    blocksize=1024)
        self._flux.start()

    def arreter(self):
        if self._flux is not None:
            self._flux.stop()
            self._flux.close()

    def lire(self, n):
        with self._verrou:
            return self._tampon[-n:].copy()


# --------------------------------------------------------------------------
# Analyse d'un bloc
# --------------------------------------------------------------------------
@dataclass
class Resultat:
    frequences: np.ndarray = field(default_factory=lambda: np.zeros(2))
    amplitudes: np.ndarray = field(default_factory=lambda: np.zeros(2))
    niveau: float = 0.0
    f0: float = None          # None = silence ou son indéterminé
    note: dict = None


class Accordeur:
    """Enchaîne les étapes de dsp.py et lisse les résultats dans le temps."""

    def __init__(self, source, a4=440.0, taille=TAILLE_BLOC):
        self.source, self.a4, self.taille = source, a4, taille
        self._historique = deque(maxlen=5)     # 5 dernières valeurs de f0

    def analyser(self):
        bloc = self.source.lire(self.taille)
        res = Resultat(niveau=niveau_rms(bloc))
        res.frequences, res.amplitudes = calculer_spectre(bloc, self.source.fs)
        if res.niveau < SEUIL_SILENCE:         # silence : on n'invente pas de note
            self._historique.clear()
            return res
        f0 = estimer_f0(res.frequences, res.amplitudes)
        if f0 is None:
            return res
        self._historique.append(f0)
        res.f0 = float(np.median(self._historique))   # la médiane élimine les valeurs aberrantes
        res.note = freq_vers_note(res.f0, self.a4)
        return res


# --------------------------------------------------------------------------
# Affichages
# --------------------------------------------------------------------------
def afficher_terminal(accordeur, duree=None):
    """Affichage minimal en mode texte : note, fréquence, jauge d'écart."""
    debut = time.time()
    try:
        while duree is None or time.time() - debut < duree:
            r = accordeur.analyser()
            if r.note is None:
                ligne = "(silence)".ljust(70)
            else:
                c = max(-50, min(50, r.note["cents"]))
                pos = int(round((c + 50) / 100 * 40))
                jauge = ["-"] * 41
                jauge[20] = "|"
                jauge[pos] = "#"
                etat = "ACCORDÉ" if abs(c) <= TOLERANCE_CENTS else ("trop bas" if c < 0 else "trop haut")
                ligne = (f"{r.note['nom_fr']:<5}{r.note['nom_intl']:<4}{r.f0:8.2f} Hz  "
                         f"[{''.join(jauge)}] {c:+6.1f} cents {etat}")
            print("\r" + ligne, end="", flush=True)
            time.sleep(0.07)
    except KeyboardInterrupt:
        pass
    print()


def lancer_interface(accordeur, fichier_capture=None, delai_capture=6.0):
    """Fenêtre matplotlib : spectre en haut, jauge de justesse en bas."""
    import matplotlib
    if fichier_capture:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax_s, ax_j) = plt.subplots(2, 1, figsize=(9, 6.5),
                                     gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("Accordeur chromatique - traitement du signal (TFD)", fontsize=13)

    # Haut : spectre d'amplitude
    ligne_spectre, = ax_s.plot([], [], lw=1.2, color="tab:blue")
    marque_f0 = ax_s.axvline(0, color="tab:red", ls="--", lw=1.2, visible=False)
    ax_s.set_xlim(0, 1500)
    ax_s.set_ylim(0, 0.4)
    ax_s.set_xlabel("Fréquence (Hz)")
    ax_s.set_ylabel("Amplitude")
    ax_s.set_title("Spectre du signal capté", fontsize=10)
    ax_s.grid(alpha=0.3)

    # Bas : jauge
    ax_j.set_xlim(-50, 50)
    ax_j.set_ylim(-1, 2.0)
    ax_j.set_yticks([])
    ax_j.set_xlabel("Écart par rapport à la note juste (cents)")
    ax_j.axvspan(-TOLERANCE_CENTS, TOLERANCE_CENTS, ymax=0.5, color="tab:green", alpha=0.25)
    ax_j.axvline(0, ymax=0.5, color="gray", lw=1)
    aiguille, = ax_j.plot([0, 0], [-0.8, 0.5], lw=4, color="tab:gray", solid_capstyle="round")
    texte_note = ax_j.text(0, 1.4, "--", ha="center", va="center", fontsize=34, weight="bold")
    texte_info = ax_j.text(0, -0.45, "", ha="center", va="center", fontsize=11)
    ax_j.text(-48, 0.75, "♭ trop bas", fontsize=9, color="gray")
    ax_j.text(48, 0.75, "trop haut ♯", fontsize=9, color="gray", ha="right")

    def maj(_):
        r = accordeur.analyser()
        ligne_spectre.set_data(r.frequences, r.amplitudes)
        if r.note is None:
            marque_f0.set_visible(False)
            texte_note.set_text("--")
            texte_info.set_text("Jouez une note...")
            aiguille.set_color("tab:gray")
            aiguille.set_xdata([0, 0])
        else:
            c = max(-50.0, min(50.0, r.note["cents"]))
            juste = abs(c) <= TOLERANCE_CENTS
            marque_f0.set_xdata([r.f0, r.f0])
            marque_f0.set_visible(True)
            texte_note.set_text(f"{r.note['nom_fr']}  ({r.note['nom_intl']})")
            texte_info.set_text(f"f0 = {r.f0:.2f} Hz   |   note juste = {r.note['f_ref']:.2f} Hz   |   "
                                f"{r.note['cents']:+.1f} cents")
            aiguille.set_color("tab:green" if juste else "tab:orange")
            aiguille.set_xdata([c, c])
        return ligne_spectre, marque_f0, aiguille, texte_note, texte_info

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if fichier_capture:
        time.sleep(delai_capture)         # laisse la source simulée avancer
        for _ in range(6):                # plusieurs analyses pour remplir le lissage
            maj(None)
            time.sleep(0.05)
        fig.savefig(fichier_capture, dpi=130)
        print("Capture enregistrée :", fichier_capture)
        return
    anim = FuncAnimation(fig, maj, interval=70, blit=False, cache_frame_data=False)
    fig._anim = anim
    plt.show()


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Accordeur chromatique temps réel (TFD/FFT)")
    p.add_argument("--simulation", action="store_true", help="notes synthétiques au lieu du micro")
    p.add_argument("--terminal", action="store_true", help="affichage texte au lieu de la fenêtre")
    p.add_argument("--a4", type=float, default=440.0, help="fréquence de référence du La4 (défaut 440)")
    p.add_argument("--micro", type=int, default=None, help="numéro du périphérique d'entrée")
    p.add_argument("--liste-micros", action="store_true", help="affiche les périphériques audio et quitte")
    p.add_argument("--capture", metavar="PNG", help="enregistre une capture d'écran puis quitte (mode simulation conseillé)")
    p.add_argument("--delai-capture", type=float, default=6.0,
                   help="secondes d'attente avant la capture (défaut 6)")
    a = p.parse_args(argv)

    if a.liste_micros:
        import sounddevice as sd
        print(sd.query_devices())
        return

    source = SourceSimulee() if a.simulation else SourceMicro(peripherique=a.micro)
    source.demarrer()
    try:
        accordeur = Accordeur(source, a4=a.a4)
        if a.terminal:
            afficher_terminal(accordeur)
        else:
            lancer_interface(accordeur, a.capture, a.delai_capture)
    finally:
        source.arreter()


if __name__ == "__main__":
    main()
