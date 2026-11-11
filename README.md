# Accordeur chromatique en temps réel (TFD / FFT)

Projet d'examen Codage et Traitement du Signal (L2) : détection de notes et accordeur chromatique
basés sur la Transformée de Fourier Discrète.

Le programme écoute le micro, calcule le spectre du son, trouve la fréquence fondamentale f0**,
puis affiche la note la plus proche et l'écart de justesse en cents**.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Utilisation

```bash
python -m src.tuner                  # micro + fenêtre graphique (spectre + jauge)
python -m src.tuner --terminal       # micro + affichage dans le terminal
python -m src.tuner --simulation     # sans micro : notes synthétiques (pour tester)
python -m src.tuner --liste-micros   # numéros des périphériques audio
python -m src.tuner --micro 1        # choisir un micro
python -m src.tuner --a4 432         # changer la fréquence de référence du La4

# Analyse hors-ligne : produit le diagramme du spectre
python -m src.analyse_fichier mon_son.wav --sortie docs/figures/spectre_wav.png
python -m src.analyse_fichier --synthese 110 --sortie docs/figures/spectre_110.png
```

Prise de captures d'écran pour le rapport : lancez `python -m src.tuner`, jouez une note, faites une
capture d'écran et enregistrez-la sous `docs/captures/capture_micro_1.png`, `capture_micro_2.png`, ...
Elles seront insérées automatiquement dans le PDF (voir ci-dessous).

## Tests

```bash
python -m pytest -v
```

## Rapport PDF

```bash
python rapport/generer_figures.py
python rapport/generer_rapport.py     # -> rapport/Rapport_Accordeur.pdf
```

## Comment ça marche

| Étape | Fichier / fonction | Idée |
|---|---|---|
| Capture | `tuner.SourceMicro` | tampon circulaire de 8192 échantillons à 44100 Hz |
| Fenêtre | `dsp.fenetre_hann` | évite la fuite spectrale |
| Spectre | `dsp.calculer_spectre` | FFT + zero-padding, amplitudes normalisées |
| TFD "à la main" | `dsp.dft_boucles`, `dsp.dft_matrice` | la définition du cours, vérifiée contre `numpy.fft` |
| Fondamentale | `dsp.estimer_f0` | Harmonic Product Spectrum + interpolation parabolique |
| Note | `dsp.freq_vers_note` | `m = 69 + 12*log2(f/440)`, écart en cents |

## Structure

```
src/dsp.py              traitement du signal (sans micro, testable seul)
src/signaux.py          signaux synthétiques, lecture WAV, source simulée
src/tuner.py            accordeur temps réel
src/analyse_fichier.py  spectre d'un WAV
tests/test_dsp.py       tests unitaires
rapport/                génération des figures et du PDF
docs/                   figures et captures
```

## Limites

Une seule note à la fois (pas d'accords) ; plage de détection 60-1500 Hz ; latence d'environ 0,2 s.
