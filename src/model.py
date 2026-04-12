from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

# ⚠️ IMPORTANT: Après chaque entraînement (python main.py),
# mettre à jour HISTORIQUE_EXPERIMENTATIONS.md avec:
# - Les logs de sortie (Epoch X/20 ...)
# - Le diagnostic (stabilité, overfitting, underfitting?)
# - La verdict final et étapes suivantes

# =============================================================
# RAPPEL : ce qu'on veut construire
#
# Entrée  → image IRM de taille (128, 128, 3)
# Sortie  → 4 probabilités  ex: [0.03, 0.08, 0.07, 0.82]
#                                  ↑       ↑      ↑      ↑
#                               notumor  mening. pitu. glioma
# =============================================================

IMG_SIZE    = 128
NUM_CLASSES = 4
# Nombre maximal d'epochs d'entraînement.
# EarlyStopping peut arrêter avant si validation n'améliore plus.
# Passé de 15 à 20 car sans flip horizontal, le modèle a besoin de plus de temps.
MAX_EPOCHS  = 20
BATCH_SIZE  = 32
USE_AUGMENTATION = False
MODEL_VERSION = "EXP4_capacite_plus_haute"


# Augmentation optionnelle (active uniquement si USE_AUGMENTATION=True).
# IMPORTANT: un Sequential vide provoque une erreur Keras.
# Donc on crée le pipeline seulement quand l'augmentation est activée.
if USE_AUGMENTATION:
    DATA_AUGMENTATION = Sequential([
        RandomRotation(0.08),
        RandomZoom(0.10),
    ], name="data_augmentation")
else:
    DATA_AUGMENTATION = None


def construire_modele():
    """
    Construit et retourne le CNN baseline décrit dans le CdC.

    Version simplifiée pour débutant:
    1. on prépare l'image
    2. on extrait des motifs simples puis complexes
    3. on transforme ces motifs en décision finale
    """

    # Sequential = on empile les couches une par une, dans l'ordre.
    # C'est la façon la plus simple de construire un réseau.
    # Chaque couche reçoit la sortie de la précédente.
    model = Sequential(name=f"CNN_Brain_Tumor_{MODEL_VERSION}")
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    # =========================================================
    # BLOC 1 : première Conv2D + MaxPooling
    # =========================================================

    # Conv2D — la couche qui "voit" l'image.
    # Elle cherche des petits motifs comme des bords, des contours
    # ou des textures utiles pour reconnaître une tumeur.
    #
    # 32      : nombre de filtres. Chaque filtre cherche un pattern différent.
    #           Avec 32 filtres, on cherche 32 choses différentes dans l'image
    #           (bords horizontaux, verticaux, zones claires, zones sombres...)
    #           Le réseau décide SEUL ce que chaque filtre va chercher pendant l'entraînement.
    #
    # (3, 3)  : taille de chaque filtre. Le filtre se déplace sur l'image
    #           en regardant 3x3 pixels à la fois (comme une loupe 3x3).
    #
    # activation='relu' : après le calcul, on met à zéro tous les résultats négatifs.
    #           Pourquoi ? Les valeurs négatives représentent "ce pattern n'est PAS là".
    #           On veut juste savoir CE QUI EST présent, pas ce qui est absent.
    #           ReLU = max(0, x) — simple mais très efficace.
    #
    if DATA_AUGMENTATION is not None:
        model.add(DATA_AUGMENTATION)
    # EXP4: on augmente un peu la capacite (32 -> 48) puis BatchNorm
    # pour stabiliser les activations entre les mini-batchs.
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    # MaxPooling2D — la couche qui "résume".
    # Elle garde l'information la plus forte dans chaque zone.
    #
    # pool_size=(2, 2) : on divise l'image en blocs de 2x2 pixels
    #           et on garde seulement le pixel le plus "activé" (le maximum).
    #           Résultat : l'image est divisée par 2 en largeur et en hauteur.
    #           128x128 → 63x63 (après Conv qui enlève 2 pixels de bordure)
    #
    # POURQUOI ? Deux raisons :
    #   1. Réduire la taille = moins de calculs → plus rapide
    #   2. Rendre le réseau robuste aux petits décalages :
    #      si une tumeur est 2 pixels à gauche, MaxPooling s'en fiche,
    #      il voit quand même le même signal fort.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =========================================================
    # BLOC 2 : deuxième Conv2D + MaxPooling
    # (même logique, mais 64 filtres au lieu de 32)
    # =========================================================

    # 64 filtres cette fois.
    # Ici on passe à un niveau de lecture plus riche: le réseau combine
    # les petits motifs du premier bloc pour repérer des formes plus utiles.
    # POURQUOI plus que la première couche ?
    # La 1ère couche détecte des choses simples (bords, textures).
    # La 2ème couche COMBINE ces détections pour voir des choses plus complexes
    # (formes, masses, contours de tumeurs).
    # Plus de complexité → besoin de plus de filtres pour tout capturer.
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Troisième bloc convolutionnel: on extrait encore plus de détails.
    # Plus on avance, plus le réseau apprend des structures abstraites.
    model.add(Conv2D(192, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =========================================================
    # TRANSITION : du monde 2D au monde 1D
    # =========================================================

    # GlobalAveragePooling2D — résume chaque carte de caractéristiques.
    # Au lieu d'aplatir tout le volume, on garde une moyenne par canal.
    # Cela réduit fortement le nombre de paramètres et aide contre l'overfitting.
    model.add(GlobalAveragePooling2D())

    # =========================================================
    # BLOC 3 : couches de décision (Dense)
    # =========================================================

    # Dropout — la couche qui empêche la mémorisation trop précise.
    # Pendant l'entraînement, elle coupe une partie des neurones au hasard.
    #
    # Pendant l'entraînement, Dropout éteint ALÉATOIREMENT 30% des neurones
    # à chaque passage (rate=0.3 → 30% éteints).
    #
    # POURQUOI ? Sans Dropout, le réseau risque de mémoriser les images
    # d'entraînement au lieu d'apprendre à généraliser.
    # C'est comme réviser un examen : si tu mémorises les questions par cœur
    # sans comprendre, tu échoues sur de nouvelles questions.
    # Dropout force le réseau à ne pas dépendre d'un seul neurone.
    #
    # NOTE : Dropout est actif SEULEMENT pendant l'entraînement.
    # Pendant la prédiction, tous les neurones sont actifs.
    # CHANGEMENT: 0.3 -> 0.5 pour plus de regularisation et stabilite.
    model.add(Dropout(0.5))

    # Dense(128) — la couche d'interprétation.
    # Elle mélange tout ce que le réseau a vu avant pour produire une idée finale.
    #
    # 128 neurones qui reçoivent chacun les valeurs du GlobalAveragePooling.
    # Chaque neurone fait une somme pondérée : il combine toutes les infos
    # et produit UN résultat. Les 128 résultats représentent 128 opinions
    # sur ce que l'image contient.
    # Ces poids (pondérations) sont ce que le réseau APPREND pendant l'entraînement.
    #
    # kernel_regularizer=l2(1e-4) : ajoute une penalite si les poids deviennent trop gros.
    # Cela force le modele a rester humble et generaliser au lieu de surspe cialiser.
    model.add(Dense(192, activation='relu', kernel_regularizer=l2(1e-4)))

    # Dropout renforce avant la sortie finale.
    model.add(Dropout(0.5))

    # Dense(NUM_CLASSES) — la couche de sortie.
    # Chaque neurone correspond a une classe: une sortie par type de tumeur.
    #
    # 4 neurones = 4 classes (notumor, meningioma, pituitary, glioma).
    # activation=softmax : transforme les 4 scores bruts en 4 probabilites
    # qui s'additionnent TOUJOURS a 1.0.
    #
    # Exemple de sortie brute avant softmax : [-1.2, 0.3, 0.1, 2.8]
    # Apres softmax                         : [0.03, 0.13, 0.11, 0.73]
    #                                          somme = 1.0
    model.add(Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(1e-4)))

    # =========================================================
    # COMPILATION — on configure l'apprentissage
    # =========================================================

    # optimizer='adam' : l'algorithme qui ajuste les poids du réseau.
    # Adam est "adaptatif" : il ajuste son taux d'apprentissage
    # automatiquement selon la progression. C'est le meilleur choix par défaut.
    # learning_rate=0.001 : à chaque erreur, on corrige de 0.1%.
    # Trop grand → le réseau oscille et n'apprend pas.
    # Trop petit → l'apprentissage est très lent.
    # 0.001 est la valeur standard pour Adam.

    # loss='categorical_crossentropy' : mesure à quel point la prédiction
    # est éloignée de la réalité.
    # On utilise "categorical" parce que nos labels sont en one-hot.
    # Ex: vrai label=[0,0,0,1], prédiction=[0.03,0.08,0.07,0.82]
    # → l'erreur est faible (bonne prédiction).
    # Ex: vrai label=[0,0,0,1], prédiction=[0.70,0.10,0.10,0.10]
    # → l'erreur est grande (mauvaise prédiction).

    # metrics=['accuracy'] : pendant l'entraînement, affiche le pourcentage
    # d'images correctement classées. C'est notre indicateur de progression.
    # CHANGEMENT: learning_rate reduit de 0.001 vers 0.0005
    # Cela rend l'apprentissage plus stable quand les donnees sont limitees.
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def afficher_architecture(model):
    """
    Affiche un résumé lisible de toutes les couches.
    Très utile pour vérifier que tout est correct.

    Quand on débute, ce résumé sert à vérifier trois choses:
    - l'ordre des couches
    - la taille des sorties
    - le nombre de paramètres à apprendre
    """
    print("\n" + "=" * 55)
    print("ARCHITECTURE DU MODÈLE")
    print("=" * 55)
    model.summary()

    # Calcul manuel du nombre de paramètres
    # Un "paramètre" = un poids que le réseau doit apprendre
    total = model.count_params()
    print(f"\nTotal paramètres à apprendre : {total:,}")
    print(f"Mémoire estimée du modèle    : ~{total * 4 / 1e6:.1f} MB")
    print("=" * 55)


def entrainer_modele(model, X_train, y_train, X_test, y_test):
    """
    Lance l'entraînement et retourne l'historique des métriques.

    Un 'epoch' = le réseau a vu TOUTES les images d'entraînement une fois.
    Avec 30 epochs, il verra chaque image 30 fois.
    À chaque fois, il améliore un peu ses poids.

    Ici on limite le nombre maximal d'epochs et on laisse EarlyStopping
    décider quand s'arrêter si la validation ne progresse plus.
    """

    print("\n" + "=" * 55)
    print("ENTRAÎNEMENT")
    print("=" * 55)
    print(f"Images train  : {X_train.shape[0]}")
    print(f"Images test   : {X_test.shape[0]}")
    print(f"Epochs (max)  : {MAX_EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print("Augmentation  : ❌ AUCUNE (EXPÉRIENCE 3: test baseline rescale seul)")
    print("               voir HISTORIQUE_EXPERIMENTATIONS.md")
    print("Early stop    : monitor=val_loss, patience=4")
    print("LR scheduler  : ReduceLROnPlateau (réduit de 50% après 2 epochs sans amélioration)")
    print("\n💡 Si validation baisse rapidement: c'est du UNDERFITTING,")
    print("   le modèle ne capte pas assez d'infos pour généraliser.")
    print()

    # batch_size=32 : le réseau ne voit pas 5600 images d'un coup.
    # Il les traite par paquets de 32 (un "batch").
    # Pourquoi ? Trois raisons :
    #   1. La RAM ne peut pas toujours tout charger en même temps.
    #   2. Traiter par batch rend l'apprentissage plus stable et plus rapide.
    #   3. Les mises à jour de poids se font plus souvent.
    # 5600 images ÷ 32 = 175 batches par epoch.

    # validation_data=(X_test, y_test) : après chaque epoch,
    # le réseau est testé sur les images de test (qu'il n'a JAMAIS vues).
    # Ça nous permet de voir si le réseau généralise vraiment
    # ou s'il mémorise juste les données d'entraînement.
    #
    # Indicateurs à surveiller:
    # - Si train_loss baisse mais val_loss remonte → OVERFITTING (trop surapprentissage)
    # - Si train_loss baisse mais val_loss aussi baisse lentement → BON SIGNE
    # - Si val_loss explose dès les premiers epochs → UNDERFITTING (le modèle ne comprend pas)

    # EarlyStopping arrête l'entraînement si la val_loss ne s'améliore plus.
    # restore_best_weights=True recharge automatiquement les meilleurs poids.
    #
    # ReduceLROnPlateau (NOUVEAU) : si la validation stagne pendant N epochs,
    # on réduit le learning rate automatiquement (ex: 0.0005 → 0.00025).
    # Cela permet au modèle d'affiner ses poids petit à petit, moins agressivement.
    # C'est comme passer d'une grosse lime à un papier de verre fin.
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,          # multiplie le learning rate par 0.5 (le divise par 2)
            patience=2,          # après 2 epochs sans amélioration
            min_lr=1e-6,         # n'ira pas en dessous de 0.000001
            verbose=1
        )
    ]

    historique = model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1    # affiche la progression à chaque epoch
    )

    return historique


def afficher_courbes(historique):
    """
    Affiche les courbes d'accuracy et de loss pendant l'entraînement.

    Ces courbes sont LA chose la plus importante à regarder après l'entraînement.
    Elles te disent si le modèle apprend bien ou s'il y a un problème.

    Si train monte et validation stagne ou descend, il y a souvent overfitting.

    GUIDE DE DIAGNOSTIC:
    ====================
    1️⃣ OVERFITTING (train bon, val mauvais):
       - train_accuracy monte jusqu'à >90%
       - val_accuracy stagne ou baisse
       - train_loss descend très bas
       - val_loss remonte
       → Solution: plus de dropout, moins d'epochs, ou data augmentation

    2️⃣ UNDERFITTING (les deux montent mais lentement):
       - train_accuracy monte lentement <70%
       - val_accuracy suit mais reste basse <65%
       - val_loss reste haute
       → Solution: modèle trop simple, ou mauvaise augmentation

    3️⃣ BON APPRENTISSAGE:
       - Les deux courbes montent ensemble
       - L'écart train/val reste <10%
       - train_loss et val_loss descendent ensemble
    """

    # On prépare deux graphiques: un pour l'accuracy, un pour la loss.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    epochs = range(1, len(historique.history['accuracy']) + 1)

    # --- Courbe 1 : Accuracy ---
    # Train accuracy : performance sur les images d'entraînement
    # Val accuracy   : performance sur les images de test (jamais vues)
    #
    # CE QU'ON VEUT VOIR : les deux courbes montent ensemble
    # PROBLÈME (overfitting) : train monte, val stagne ou descend
    #   → le réseau mémorise au lieu d'apprendre
    axes[0].plot(epochs, historique.history['accuracy'],
                 label='Train', linewidth=2, color='#378ADD')
    axes[0].plot(epochs, historique.history['val_accuracy'],
                 label='Validation', linewidth=2, color='#E24B4A',
                 linestyle='--')
    axes[0].set_title('Accuracy par epoch', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # --- Courbe 2 : Loss (erreur) ---
    # CE QU'ON VEUT VOIR : les deux courbes descendent ensemble
    # PROBLÈME (overfitting) : train descend, val remonte
    axes[1].plot(epochs, historique.history['loss'],
                 label='Train', linewidth=2, color='#378ADD')
    axes[1].plot(epochs, historique.history['val_loss'],
                 label='Validation', linewidth=2, color='#E24B4A',
                 linestyle='--')
    axes[1].set_title('Loss (erreur) par epoch', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('courbes_entrainement.png', dpi=100)
    plt.show()
    print("Courbes sauvegardées : courbes_entrainement.png")


def sauvegarder_modele(model, chemin='models/cnn_baseline.keras'):
    """
    Sauvegarde le modèle entraîné sur le disque.

    POURQUOI sauvegarder ?
    L'entraînement peut prendre 10-30 minutes.
    Si tu sauvegardes, tu peux recharger le modèle en 1 seconde
    et faire des prédictions sans ré-entraîner.
    """
    import os
    os.makedirs('models', exist_ok=True)
    model.save(chemin)
    print(f"\nModèle sauvegardé : {chemin}")
    print("Pour le recharger plus tard :")
    print("  from tensorflow.keras.models import load_model")
    print(f"  model = load_model('{chemin}')")
