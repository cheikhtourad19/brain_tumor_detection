import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
NUM_CLASSES = 4

CLASS_MAPPING = {
    'notumor':    0,
    'meningioma': 1,
    'pituitary':  2,
    'glioma':     3
}

CACHE_DIR = os.path.join('data', 'cache_tl')

CACHE_FILES = {
    'X_train':     os.path.join(CACHE_DIR, 'X_train.npy'),
    'y_train':     os.path.join(CACHE_DIR, 'y_train.npy'),
    'y_train_raw': os.path.join(CACHE_DIR, 'y_train_raw.npy'),
    'X_val':       os.path.join(CACHE_DIR, 'X_val.npy'),
    'y_val':       os.path.join(CACHE_DIR, 'y_val.npy'),
    'X_test':      os.path.join(CACHE_DIR, 'X_test.npy'),
    'y_test':      os.path.join(CACHE_DIR, 'y_test.npy'),
    'y_test_raw':  os.path.join(CACHE_DIR, 'y_test_raw.npy'),
}


def charger_image(chemin_image):
    image = cv2.imread(chemin_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype('float32')
    return image


def charger_dataset(chemin_dossier):
    images = []
    labels = []

    print(f"\n  Lecture depuis : {chemin_dossier}")

    for nom_classe in os.listdir(chemin_dossier):
        if nom_classe not in CLASS_MAPPING:
            continue

        label_numerique = CLASS_MAPPING[nom_classe]
        chemin_classe = os.path.join(chemin_dossier, nom_classe)
        fichiers = os.listdir(chemin_classe)

        print(f"    '{nom_classe}' (label={label_numerique}) : {len(fichiers)} images")

        for nom_fichier in fichiers:
            if not nom_fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            chemin_complet = os.path.join(chemin_classe, nom_fichier)
            images.append(charger_image(chemin_complet))
            labels.append(label_numerique)

    X = np.array(images, dtype='float32')
    y = np.array(labels, dtype='int32')
    X, y = shuffle(X, y, random_state=42)
    y_onehot = to_categorical(y, num_classes=NUM_CLASSES)

    return X, y_onehot, y


def obtenir_donnees(chemin_train, chemin_test, forcer_recalcul=True):
    cache_complet = all(os.path.exists(f) for f in CACHE_FILES.values())

    if cache_complet and not forcer_recalcul:
        print("Cache TL trouvé — chargement rapide...")
        X_train     = np.load(CACHE_FILES['X_train'])
        y_train     = np.load(CACHE_FILES['y_train'])
        y_train_raw = np.load(CACHE_FILES['y_train_raw'])
        X_val       = np.load(CACHE_FILES['X_val'])
        y_val       = np.load(CACHE_FILES['y_val'])
        X_test      = np.load(CACHE_FILES['X_test'])
        y_test      = np.load(CACHE_FILES['y_test'])
        y_test_raw  = np.load(CACHE_FILES['y_test_raw'])

        _afficher_resume(X_train, y_train, X_val, y_val, X_test, y_test)
        return X_train, y_train, y_train_raw, X_val, y_val, X_test, y_test, y_test_raw

    if forcer_recalcul:
        print("Recalcul forcé — prétraitement TL en cours...")

    print("\nTraitement du Training set...")
    X_train_complet, y_train_complet, y_train_complet_raw = charger_dataset(chemin_train)

    print("\nCréation du split train / validation (80% / 20%)...")
    X_train, X_val, y_train, y_val, y_train_raw, _ = train_test_split(
        X_train_complet,
        y_train_complet,
        y_train_complet_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_train_complet_raw
    )

    print(f"  X_train : {X_train.shape[0]} images")
    print(f"  X_val   : {X_val.shape[0]} images")

    print("\nTraitement du Testing set...")
    X_test, y_test, y_test_raw = charger_dataset(chemin_test)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"\nSauvegarde du cache TL dans '{CACHE_DIR}/'...")
    np.save(CACHE_FILES['X_train'],     X_train)
    np.save(CACHE_FILES['y_train'],     y_train)
    np.save(CACHE_FILES['y_train_raw'], y_train_raw)
    np.save(CACHE_FILES['X_val'],       X_val)
    np.save(CACHE_FILES['y_val'],       y_val)
    np.save(CACHE_FILES['X_test'],      X_test)
    np.save(CACHE_FILES['y_test'],      y_test)
    np.save(CACHE_FILES['y_test_raw'],  y_test_raw)
    print("  Sauvegarde terminée.")

    _afficher_resume(X_train, y_train, X_val, y_val, X_test, y_test)
    return X_train, y_train, y_train_raw, X_val, y_val, X_test, y_test, y_test_raw


def _afficher_resume(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 48)
    print("DONNÉES TL PRÊTES")
    print("=" * 48)
    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print("=" * 48)