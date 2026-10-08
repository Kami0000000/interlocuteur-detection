# Reconnaissance du Locuteur — Déverrouillage Vocal

## Description
Système de reconnaissance du locuteur basé sur un MLP simple (sans CNN/RNN).
Simule un déverrouillage vocal de téléphone : le propriétaire déverrouille,
les autres voix sont rejetées.

## Architecture
- **Type** : MLP (Multi-Layer Perceptron)
- **Couches** : Dense(512) → Dropout → Dense(256) → Dropout → Dense(128) → Dense(6, softmax)
- **Features** : 40 coefficients MFCC
- **Classes** : 5 politiciens + AUTRES
- **Paramètres** : 185 990 (~2 Mo)

## Performances
- Précision : **99,67%** (sur 3896 fichiers jamais vus)
- FAR (faux positifs) : 0,03%
- FRR (faux négatifs) : 0,00%

## Fichiers
- `Script/fit.py` : Entraînement
- `Script/test.py` : Test
- `Modele/modele_enrichi.keras` : Modèle entraîné
- `Modele/scaler_enrichi.pkl` : Normalisation
