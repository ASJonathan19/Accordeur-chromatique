"""
analyse_fichier.py - Analyse hors-ligne d'un fichier WAV (ou d'une note synthétique).

    python -m src.analyse_fichier mon_son.wav --sortie docs/figures/spectre_wav.png
    python -m src.analyse_fichier --synthese 110 --sortie docs/figures/spectre_110.png

Produit le diagramme du spectre fréquentiel demandé dans l'énoncé et affiche f0 et la note.
"""
import argparse

import numpy as np

from .dsp import calculer_spectre, estimer_f0, freq_vers_note
from .signaux import lire_wav, note_synthetique

TAILLE_BLOC = 8192


def analyser(signal, fs, debut=None):
    """Analyse un bloc de TAILLE_BLOC échantillons pris à 'debut' (par défaut au milieu)."""
    if len(signal) < TAILLE_BLOC:
        signal = np.pad(signal, (0, TAILLE_BLOC - len(signal)))
    if debut is None:
        debut = max(0, (len(signal) - TAILLE_BLOC) // 2)
    bloc = signal[debut:debut + TAILLE_BLOC]
    freqs, amps = calculer_spectre(bloc, fs)
    f0 = estimer_f0(freqs, amps)
    return bloc, freqs, amps, f0


def tracer(bloc, freqs, amps, f0, fs, chemin, fmax=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    t = np.arange(len(bloc)) / fs * 1000
    a1.plot(t[:int(0.04 * fs)], bloc[:int(0.04 * fs)], lw=1)
    a1.set_xlabel("Temps (ms)"); a1.set_ylabel("Amplitude")
    a1.set_title("Signal capté (40 premières ms du bloc)"); a1.grid(alpha=0.3)
    a2.plot(freqs, amps, lw=1.2)
    if f0:
        n = freq_vers_note(f0)
        a2.axvline(f0, color="red", ls="--", label=f"f0 = {f0:.2f} Hz -> {n['nom_intl']} ({n['cents']:+.1f} cents)")
        for h in range(2, 6):
            if h * f0 < fmax:
                a2.axvline(h * f0, color="gray", ls=":", lw=0.8)
        a2.legend()
    a2.set_xlim(0, fmax)
    a2.set_xlabel("Fréquence (Hz)"); a2.set_ylabel("Amplitude")
    a2.set_title("Spectre d'amplitude (fenêtre de Hann + FFT)"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyse d'un WAV ou d'une note synthétique")
    p.add_argument("wav", nargs="?", help="fichier WAV PCM 16 bits")
    p.add_argument("--synthese", type=float, metavar="HZ", help="générer une note de fréquence HZ")
    p.add_argument("--sortie", default="spectre.png", help="image du spectre")
    a = p.parse_args(argv)
    if a.wav:
        x, fs = lire_wav(a.wav)
    elif a.synthese:
        fs = 44100
        x = note_synthetique(a.synthese, 0.6, fs)
    else:
        p.error("donnez un fichier WAV ou --synthese HZ")
    bloc, freqs, amps, f0 = analyser(x, fs)
    if f0 is None:
        print("Aucune fréquence fondamentale détectée.")
    else:
        n = freq_vers_note(f0)
        print(f"f0 = {f0:.2f} Hz  ->  {n['nom_fr']} ({n['nom_intl']}), écart {n['cents']:+.1f} cents")
    tracer(bloc, freqs, amps, f0, fs, a.sortie)
    print("Spectre enregistré :", a.sortie)


if __name__ == "__main__":
    main()
