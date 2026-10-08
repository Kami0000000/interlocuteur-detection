import numpy as np
import tensorflow as tf
import librosa
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# =========================================================
# PARAMÈTRES
# =========================================================

TAILLE_MFCC = 40
SAMPLE_RATE = 16000
DUREE_AUDIO = 3.0

CLASSES_POLITICIENS = ['Benjamin_Netanyau', 'Jens_Stoltenberg', 'Julia_Gillard',
                        'Magaret_Tarcher', 'Nelson_Mandela']
CLASSES_TOTALES = CLASSES_POLITICIENS + ['AUTRES']
PROPRIETAIRE = 'Nelson_Mandela'
INDEX_PROPRIETAIRE = CLASSES_TOTALES.index(PROPRIETAIRE)
LABEL_AUTRES = 5

BASE = "/home/kami/Documents/Neurone/Interlocuteur"
dossier_politiciens = os.path.join(BASE, "Donnee/archive/16000_pcm_speeches")
dossier_hindi      = os.path.join(BASE, "Donnee/archive/speaker-recognition-dataset/hindi")
chemin_modele      = os.path.join(BASE, "Modele/modele_6classes.h5")
chemin_scaler      = os.path.join(BASE, "Modele/scaler.pkl")

# =========================================================
# CHARGEMENT
# =========================================================

modele = tf.keras.models.load_model(chemin_modele)
scaler = joblib.load(chemin_scaler)
print("✅ Modèle et scaler chargés.")

def extraire_mfcc(chemin_audio):
    try:
        y, sr = librosa.load(chemin_audio, sr=SAMPLE_RATE, duration=DUREE_AUDIO)
        if len(y) < sr * 0.5:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=TAILLE_MFCC)
        return np.mean(mfcc, axis=1)
    except:
        return None

# =========================================================
# RECONSTRUCTION DU DATASET
# =========================================================

print("\nReconstruction du dataset complet...")

X, y = [], []

for idx, classe in enumerate(CLASSES_POLITICIENS):
    chemin_classe = os.path.join(dossier_politiciens, classe)
    fichiers = sorted(glob.glob(os.path.join(chemin_classe, "**", "*.wav"), recursive=True))
    for f in fichiers:
        feat = extraire_mfcc(f)
        if feat is not None:
            X.append(feat)
            y.append(idx)

fichiers_hindi = sorted(glob.glob(os.path.join(dossier_hindi, "**", "*.wav"), recursive=True))
for f in fichiers_hindi:
    feat = extraire_mfcc(f)
    if feat is not None:
        X.append(feat)
        y.append(LABEL_AUTRES)

X = np.array(X)
y = np.array(y)
print(f"Total fichiers chargés : {len(X)}")
print(f"Répartition : {np.bincount(y)}")

# =========================================================
# SPLIT (MÊME QUE PENDANT L'ENTRAÎNEMENT)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nX_test : {len(X_test)} fichiers")
print(f"Répartition X_test : {np.bincount(y_test)}")

# =========================================================
# ÉVALUATION
# =========================================================

X_test_norm = scaler.transform(X_test)
probas = modele.predict(X_test_norm, verbose=0)
predictions = np.argmax(probas, axis=1)

accuracy = accuracy_score(y_test, predictions)
print(f"\n{'='*60}")
print(f"✅ PRÉCISION RÉELLE SUR DONNÉES INÉDITES : {accuracy*100:.2f}%")
print(f"{'='*60}")

# Détail par classe
print("\n📊 Détail par classe :")
for idx, nom in enumerate(CLASSES_TOTALES):
    masque = (y_test == idx)
    total = masque.sum()
    if total > 0:
        correct = (predictions[masque] == idx).sum()
        print(f"  {nom:<20}: {correct}/{total} ({correct/total*100:.1f}%)")
    else:
        print(f"  {nom:<20}: 0 fichier dans X_test (pas d'évaluation possible)")

# Matrice de confusion CORRIGÉE (avec labels explicites)
print("\n📊 Matrice de confusion (lignes=vrai, colonnes=prédit) :")
cm = confusion_matrix(y_test, predictions, labels=range(len(CLASSES_TOTALES)))

# En-tête
print(f"{'':<20}", end="")
for nom in CLASSES_TOTALES:
    print(f"{nom[:8]:>10}", end="")
print()
# Lignes
for i, nom in enumerate(CLASSES_TOTALES):
    print(f"{nom:<20}", end="")
    for j in range(len(CLASSES_TOTALES)):
        print(f"{cm[i][j]:>10}", end="")
    print()

# =========================================================
# DÉCISION DÉVERROUILLAGE
# =========================================================

print(f"\n{'='*60}")
print("DÉCISION DE DÉVERROUILLAGE (données inédites)")
print(f"{'='*60}")

vp = fp = vn = fn = 0
for vrai, predit in zip(y_test, predictions):
    if vrai == INDEX_PROPRIETAIRE and predit == INDEX_PROPRIETAIRE:
        vp += 1
    elif vrai != INDEX_PROPRIETAIRE and predit == INDEX_PROPRIETAIRE:
        fp += 1
    elif vrai != INDEX_PROPRIETAIRE and predit != INDEX_PROPRIETAIRE:
        vn += 1
    elif vrai == INDEX_PROPRIETAIRE and predit != INDEX_PROPRIETAIRE:
        fn += 1

print(f"\n🔓 Propriétaire reconnu (VP)            : {vp}")
print(f"🔓 Non-propriétaire déverrouillé (FP)   : {fp}")
print(f"🔒 Non-propriétaire verrouillé (VN)     : {vn}")
print(f"🔒 Propriétaire verrouillé (FN)         : {fn}")

if (fp + vn) > 0:
    print(f"\n📊 FAR (Faux Accept Rate) : {fp/(fp+vn)*100:.2f}%")
if (fn + vp) > 0:
    print(f"📊 FRR (Faux Reject Rate) : {fn/(fn+vp)*100:.2f}%")
