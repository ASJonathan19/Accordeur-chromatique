"""
dsp.py - Briques de traitement du signal de l'accordeur chromatique.

Contenu :
    1. dft_boucles()  : la TFD "à la main" (double boucle, comme dans le cours)
    2. dft_matrice()  : la même TFD écrite avec une matrice numpy
    3. fenetre_hann() : fenêtre d'apodisation
    4. calculer_spectre() : fenêtre + FFT -> spectre d'amplitude
    5. estimer_f0()   : extraction de la fréquence fondamentale (HPS + interpolation)
    6. freq_vers_note() : fréquence -> nom de la note + écart en cents

Aucune dépendance au micro ni à l'affichage : ce module est testable seul.
"""
import cmath
import math

import numpy as np

NOTES_INTL = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_FR = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


# --------------------------------------------------------------------------
# 1. La Transformée de Fourier Discrète
# --------------------------------------------------------------------------
def dft_boucles(x):
    """TFD par définition : X[k] = somme_{n=0}^{N-1} x[n] * exp(-i*2*pi*k*n/N).

    C'est exactement le calcul fait à la main dans le cours (tableau k, n).
    Complexité O(N^2) : très lent dès que N dépasse quelques milliers.
    """
    N = len(x)
    X = [0j] * N                      # X = 0, 0, ..., 0  (N valeurs)
    for k in range(N):                # pour k de 0 à N-1
        for n in range(N):            # pour n de 0 à N-1
            X[k] += x[n] * cmath.exp(-2j * math.pi * k * n / N)
    return np.array(X)


def dft_matrice(x):
    """Même calcul, mais vectorisé : X = W @ x, avec W[k, n] = exp(-i*2*pi*k*n/N)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x


# --------------------------------------------------------------------------
# 2. Fenêtre et spectre
# --------------------------------------------------------------------------
def fenetre_hann(N):
    """Fenêtre de Hann (version périodique) : 0.5 - 0.5*cos(2*pi*n/N).

    Elle ramène doucement le signal à 0 aux deux bords du bloc. Sans elle,
    la TFD "voit" une coupure brutale et l'énergie de chaque note se répand
    sur toutes les fréquences (fuite spectrale).
    """
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)


def calculer_spectre(bloc, fs, zero_padding=2):
    """Retourne (frequences, amplitudes) du bloc audio.

    bloc         : tableau 1D de N échantillons
    fs           : fréquence d'échantillonnage (Hz)
    zero_padding : on complète avec des zéros jusqu'à N*zero_padding points ;
                   cela ne crée pas d'information mais "lisse" le spectre
                   (plus de points -> pic mieux localisé).
    """
    bloc = np.asarray(bloc, dtype=float)
    N = len(bloc)
    fen = fenetre_hann(N)
    x = (bloc - bloc.mean()) * fen            # on retire la composante continue
    n_fft = N * zero_padding
    X = np.fft.rfft(x, n=n_fft)               # FFT = algorithme rapide de la TFD
    amplitudes = np.abs(X) * 2 / fen.sum()    # normalisation : sinus d'amplitude A -> pic ~ A
    frequences = np.fft.rfftfreq(n_fft, d=1 / fs)
    return frequences, amplitudes


# --------------------------------------------------------------------------
# 3. Fréquence fondamentale
# --------------------------------------------------------------------------
def spectre_produit_harmonique(amplitudes, nb_harmoniques=4):
    """Harmonic Product Spectrum (HPS), calculé en logarithme (somme au lieu de produit).

    HPS[k] = log A[k] + log A[2k] + log A[3k] + ... + log A[H*k]
    Si f0 est la fondamentale, les harmoniques 2*f0, 3*f0... sont aussi des
    pics : leurs contributions s'additionnent à l'indice k0 de f0, et à aucun
    autre indice -> le maximum du HPS désigne f0 (et évite de confondre
    f0 avec son 2e harmonique, souvent plus fort sur une guitare).
    """
    log_a = np.log(amplitudes + 1e-12)
    L = len(log_a) // nb_harmoniques
    hps = np.zeros(L)
    for h in range(1, nb_harmoniques + 1):
        hps += log_a[::h][:L]          # log_a[::h][k] == log_a[h*k]
    return hps


def _interpolation_parabolique(log_a, k):
    """Affine la position du pic : parabole passant par les 3 points autour de k.
    Retourne un indice fractionnaire (ex. 52.37) -> précision bien meilleure
    que la largeur d'un bin."""
    if k <= 0 or k >= len(log_a) - 1:
        return float(k)
    a, b, c = log_a[k - 1], log_a[k], log_a[k + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(k)
    return k + 0.5 * (a - c) / denom


def estimer_f0(frequences, amplitudes, fmin=60.0, fmax=1500.0, nb_harmoniques=4,
               seuil_fondamentale=0.05):
    """Estime la fréquence fondamentale f0 (Hz) ou None si rien de net.

    Étapes :
        1. HPS sur la plage [fmin, fmax] -> indice candidat k0
        2. On se recale sur le pic réel du spectre à +/- 2 bins
        3. Garde-fou : si le candidat est ridiculement faible devant le plus
           grand pic (cas d'un son quasi sinusoïdal), on prend le plus grand pic
        4. Interpolation parabolique -> f0 précise
    """
    df = frequences[1] - frequences[0]
    hps = spectre_produit_harmonique(amplitudes, nb_harmoniques)
    k_min = max(int(math.ceil(fmin / df)), 2)
    k_max = min(int(math.floor(fmax / df)), len(hps) - 1)
    if k_max <= k_min:
        return None

    # 1. candidat HPS
    k0 = k_min + int(np.argmax(hps[k_min:k_max + 1]))

    # 2. recalage sur le pic local du spectre
    lo, hi = max(k0 - 2, 1), min(k0 + 3, len(amplitudes) - 1)
    k0 = lo + int(np.argmax(amplitudes[lo:hi]))

    # 3. garde-fou
    k_pic = k_min + int(np.argmax(amplitudes[k_min:k_max + 1]))
    if amplitudes[k0] < seuil_fondamentale * amplitudes[k_pic]:
        k0 = k_pic

    # 4. interpolation parabolique (sur le log de l'amplitude)
    k_fin = _interpolation_parabolique(np.log(amplitudes + 1e-12), k0)
    return k_fin * df


# --------------------------------------------------------------------------
# 4. Fréquence -> note
# --------------------------------------------------------------------------
def freq_vers_note(f, a4=440.0):
    """Convertit une fréquence en note la plus proche (gamme tempérée).

    Numéro MIDI :  m = 69 + 12*log2(f / 440)   (La4 = 69)
    La note la plus proche est round(m) ; l'écart restant, en centièmes de
    demi-ton (cents), est 100 * (m - round(m)).  Un demi-ton = 100 cents.
    """
    m = 69 + 12 * math.log2(f / a4)
    n = int(round(m))
    cents = 100.0 * (m - n)
    indice = n % 12
    octave = n // 12 - 1
    return {
        "midi": n,
        "nom_intl": f"{NOTES_INTL[indice]}{octave}",
        "nom_fr": f"{NOTES_FR[indice]}{octave}",
        "f_ref": a4 * 2 ** ((n - 69) / 12),
        "cents": cents,
    }


def niveau_rms(bloc):
    """Niveau efficace du bloc (sert à détecter le silence)."""
    bloc = np.asarray(bloc, dtype=float)
    return float(np.sqrt(np.mean(bloc ** 2)))
