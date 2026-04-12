import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

# =============================================================
# CONFIGURATION CENTRALE
# =============================================================

IMG_SIZE    = 128
NUM_CLASSES = 4

CLASS_MAPPING = {
    'notumor':    0,
    'meningioma': 1,
    'pituitary':  2,
    'glioma':     3
}

# Dossier où on sauvegarde les fichiers .npy
# Créé automatiquement s'il n'existe pas
CACHE_DIR = os.path.join('data', 'cache')

# Noms des fichiers cache
CACHE_FILES = {
    'X_train': os.path.join(CACHE_DIR, 'X_train.npy'),
    'y_train': os.path.join(CACHE_DIR, 'y_train.npy'),
    'y_train_raw': os.path.join(CACHE_DIR, 'y_train_raw.npy'),
    'X_test':  os.path.join(CACHE_DIR, 'X_test.npy'),
    'y_test':  os.path.join(CACHE_DIR, 'y_test.npy'),
    'y_test_raw':  os.path.join(CACHE_DIR, 'y_test_raw.npy'),
}


# =============================================================
# FONCTIONS DE TRAITEMENT (inchangées)
# =============================================================

def charger_image(chemin_image):
    image = cv2.imread(chemin_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0
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


# =============================================================
# FONCTION PRINCIPALE : cache intelligent
#
# C'est le coeur de l'optimisation.
# Elle répond à la question : "est-ce que j'ai déjà fait
# ce travail avant ?" et agit en conséquence.
# =============================================================

def obtenir_donnees(chemin_train, chemin_test, forcer_recalcul=False):
    """
    Point d'entrée unique pour obtenir les données.
    Gère automatiquement le cache.

    Paramètres :
        chemin_train    : chemin vers data/Training
        chemin_test     : chemin vers data/Testing
        forcer_recalcul : si True, ignore le cache et refait tout
                          (utile si tu changes le dataset ou IMG_SIZE)

    Retourne : X_train, y_train, y_train_raw, X_test, y_test, y_test_raw
    """

    # --- CAS 1 : le cache existe et on ne force pas le recalcul ---
    #
    # all() retourne True seulement si TOUTES les conditions sont vraies.
    # On vérifie que chacun des 6 fichiers .npy existe bien sur le disque.
    # Si un seul manque, on refait tout (évite les données incomplètes).

    cache_complet = all(os.path.exists(f) for f in CACHE_FILES.values())

    if cache_complet and not forcer_recalcul:
        print("Cache trouvé — chargement rapide des données prétraitées...")
        print(f"  (supprime '{CACHE_DIR}/' pour forcer un recalcul)")

        X_train     = np.load(CACHE_FILES['X_train'])
        y_train     = np.load(CACHE_FILES['y_train'])
        y_train_raw = np.load(CACHE_FILES['y_train_raw'])
        X_test      = np.load(CACHE_FILES['X_test'])
        y_test      = np.load(CACHE_FILES['y_test'])
        y_test_raw  = np.load(CACHE_FILES['y_test_raw'])

        _afficher_resume(X_train, y_train, X_test, y_test)
        return X_train, y_train, y_train_raw, X_test, y_test, y_test_raw

    # --- CAS 2 : pas de cache, ou recalcul forcé ---

    if forcer_recalcul:
        print("Recalcul forcé — on ignore le cache existant.")
    else:
        print("Aucun cache trouvé — prétraitement complet en cours...")
        print("(Ce traitement ne se fera qu'une seule fois)")

    print("\nTraitement du Training set...")
    X_train, y_train, y_train_raw = charger_dataset(chemin_train)

    print("\nTraitement du Testing set...")
    X_test, y_test, y_test_raw = charger_dataset(chemin_test)

    # --- Sauvegarde du cache ---
    #
    # os.makedirs crée le dossier data/cache/ s'il n'existe pas.
    # exist_ok=True évite une erreur si le dossier existe déjà.
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"\nSauvegarde du cache dans '{CACHE_DIR}/'...")
    np.save(CACHE_FILES['X_train'],     X_train)
    np.save(CACHE_FILES['y_train'],     y_train)
    np.save(CACHE_FILES['y_train_raw'], y_train_raw)
    np.save(CACHE_FILES['X_test'],      X_test)
    np.save(CACHE_FILES['y_test'],      y_test)
    np.save(CACHE_FILES['y_test_raw'],  y_test_raw)
    print("  Sauvegarde terminée — les prochains lancements seront rapides.")

    _afficher_resume(X_train, y_train, X_test, y_test)
    return X_train, y_train, y_train_raw, X_test, y_test, y_test_raw


# =============================================================
# FONCTIONS DE VISUALISATION (inchangées)
# =============================================================

def visualiser_exemples(X, y_original, n_par_classe=3):
    noms_classes = {v: k for k, v in CLASS_MAPPING.items()}
    fig, axes = plt.subplots(NUM_CLASSES, n_par_classe,
                             figsize=(n_par_classe * 3, NUM_CLASSES * 3))
    fig.suptitle("Exemples d'images par classe", fontsize=14)

    for classe_id in range(NUM_CLASSES):
        indices = np.where(y_original == classe_id)[0]
        choisis = np.random.choice(indices, n_par_classe, replace=False)
        for col, idx in enumerate(choisis):
            ax = axes[classe_id][col]
            ax.imshow(X[idx])
            ax.set_title(f"{noms_classes[classe_id]}\nlabel={classe_id}", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('exemples_dataset.png', dpi=100, bbox_inches='tight')
    plt.show()


def visualiser_distribution(y_original, titre="Distribution des classes"):
    noms_classes = list(CLASS_MAPPING.keys())
    comptes  = np.bincount(y_original)
    couleurs = ['#639922', '#378ADD', '#EF9F27', '#E24B4A']

    plt.figure(figsize=(8, 4))
    barres = plt.bar(noms_classes, comptes, color=couleurs, edgecolor='none')
    for barre, compte in zip(barres, comptes):
        plt.text(barre.get_x() + barre.get_width() / 2,
                 barre.get_height() + 20,
                 str(compte), ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    plt.title(titre, fontsize=13)
    plt.ylabel("Nombre d'images")
    plt.ylim(0, max(comptes) * 1.2)
    plt.tight_layout()
    plt.savefig('distribution_classes.png', dpi=100)
    plt.show()


# =============================================================
# FONCTION UTILITAIRE PRIVÉE
# Le _ devant le nom indique que c'est une fonction interne,
# pas destinée à être appelée depuis main.py
# =============================================================

def _afficher_resume(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 42)
    print("DONNÉES PRÊTES")
    print("=" * 42)
    print(f"  X_train : {X_train.shape}   ~{X_train.nbytes / 1e6:.0f} MB")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}    ~{X_test.nbytes / 1e6:.0f} MB")
    print(f"  y_test  : {y_test.shape}")
    print("=" * 42)