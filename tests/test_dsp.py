"""Tests unitaires : python -m pytest -v"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dsp import (calculer_spectre, dft_boucles, dft_matrice, estimer_f0,
                     fenetre_hann, freq_vers_note)
from src.signaux import note_synthetique

FS = 44100


def test_dft_exemple_du_cours():
    x = [1, 6, 8, -4, 9]
    attendu = np.fft.fft(x)
    assert dft_boucles(x)[0] == pytest.approx(20)      # X[0] = somme des x[n]
    assert np.allclose(dft_boucles(x), attendu)
    assert np.allclose(dft_matrice(x), attendu)


def test_dft_signal_aleatoire():
    x = np.random.default_rng(3).standard_normal(64)
    assert np.allclose(dft_matrice(x), np.fft.fft(x))


def test_hann_bords_et_milieu():
    w = fenetre_hann(1024)
    assert w[0] == pytest.approx(0)
    assert w[512] == pytest.approx(1)


def test_pic_du_spectre_amplitude():
    t = np.arange(8192) / FS
    f, a = calculer_spectre(0.5 * np.sin(2 * np.pi * 1000 * t), FS)
    assert f[np.argmax(a)] == pytest.approx(1000, abs=3)
    assert a.max() == pytest.approx(0.5, rel=0.1)


@pytest.mark.parametrize("f0", [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 880.0, 1318.5])
@pytest.mark.parametrize("cents", [-30, 0, 17])
@pytest.mark.parametrize("harmoniques", [(1, .6, .4, .25, .12), (0.3, 1.0, .5, .2)])
def test_f0_precision(f0, cents, harmoniques):
    """Précision < 1 cent, y compris quand l'harmonique 2 est plus fort que la fondamentale."""
    vrai = f0 * 2 ** (cents / 1200)
    x = note_synthetique(vrai, 0.5, FS, harmoniques=harmoniques)[:8192]
    f, a = calculer_spectre(x, FS)
    mesure = estimer_f0(f, a)
    assert mesure is not None
    assert abs(1200 * math.log2(mesure / vrai)) < 1.0


def test_f0_sinus_pur_precision_3_cents():
    x = note_synthetique(82.41, 0.5, FS, harmoniques=(1.0,))[:8192]
    f, a = calculer_spectre(x, FS)
    assert abs(1200 * math.log2(estimer_f0(f, a) / 82.41)) < 3


def test_freq_vers_note():
    n = freq_vers_note(440.0)
    assert n["nom_intl"] == "A4" and n["nom_fr"] == "La4" and abs(n["cents"]) < 1e-9
    n = freq_vers_note(261.63)
    assert n["nom_intl"] == "C4" and abs(n["cents"]) < 0.1
    n = freq_vers_note(440 * 2 ** (20 / 1200))
    assert n["nom_intl"] == "A4" and n["cents"] == pytest.approx(20)
    assert freq_vers_note(82.41)["nom_intl"] == "E2"
    assert freq_vers_note(440.0, a4=432.0)["cents"] > 0      # référence 432 Hz : 440 est trop haut
