from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, RandomFlip, RandomRotation,
    RandomZoom, RandomContrast
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np


IMG_SIZE    = 128
NUM_CLASSES = 4
MAX_EPOCHS  = 30
BATCH_SIZE  = 32
USE_AUGMENTATION = True
MODEL_VERSION = "EXP5_encodeur_decodeur_dense"


if USE_AUGMENTATION:
    DATA_AUGMENTATION = Sequential([
        RandomRotation(0.08),
        RandomZoom(0.10),
        RandomFlip("horizontal"),   # IRM → OK horizontal
        RandomContrast(0.1),        # variations d'intensité
    ], name="data_augmentation")
else:
    DATA_AUGMENTATION = None


def construire_modele():
    """
    Construit et retourne le CNN avec architecture Encodeur → Décodeur → Dense.

    Encodeur  : compresse l'image (Conv + MaxPool) pour extraire les features
    Décodeur  : remonte en résolution (UpSampling + Conv) pour enrichir la repr.
    Dense     : classifie à partir du vecteur résumé (GAP → Dropout → Dense)
    """

    model = Sequential(name=f"CNN_Brain_Tumor_{MODEL_VERSION}")
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    # =========================================================
    # AUGMENTATION
    # =========================================================
    if DATA_AUGMENTATION is not None:
        model.add(DATA_AUGMENTATION)

    # =========================================================
    # ENCODEUR — compresse l'image progressivement
    # Chaque bloc réduit la taille spatiale par 2 (MaxPool)
    # et augmente la profondeur (plus de filtres = plus de features)
    # 128×128 → 64×64 → 32×32 → 16×16
    # =========================================================

    # Bloc encodeur 1 — features simples (bords, textures)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 128 → 64

    # Bloc encodeur 2 — features intermédiaires (formes, contours)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 64 → 32

    # Bloc encodeur 3 — features complexes (structures tumorales)
    # C'est le "goulot d'étranglement" : représentation la plus compacte
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 32 → 16

    # =========================================================
    # DÉCODEUR — remonte en résolution pour enrichir la représentation
    # UpSampling2D double la taille spatiale (inverse du MaxPool)
    # On réduit progressivement les filtres en remontant
    # 16×16 → 32×32 → 64×64
    # =========================================================

    # Bloc décodeur 1
    model.add(UpSampling2D(size=(2, 2)))        # 16 → 32
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Bloc décodeur 2
    model.add(UpSampling2D(size=(2, 2)))        # 32 → 64
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # =========================================================
    # DENSE — classification finale
    # GlobalAveragePooling résume chaque carte de features en un scalaire.
    # On passe d'un volume (64, 64, 32) à un vecteur de 32 valeurs.
    # =========================================================
    model.add(GlobalAveragePooling2D())

    # Dropout avant Dense : régularise et évite la mémorisation
    model.add(Dropout(0.5))

    # Couche dense principale : combine toutes les features
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))

    # Dropout final avant la sortie
    model.add(Dropout(0.4))

    # Sortie : 4 probabilités (notumor, meningioma, pituitary, glioma)
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def afficher_architecture(model):
    """
    Affiche un résumé lisible de toutes les couches.
    Vérifie : ordre des couches, taille des sorties, nombre de paramètres.
    """
    print("\n" + "=" * 55)
    print("ARCHITECTURE DU MODÈLE")
    print("=" * 55)
    model.summary()

    total = model.count_params()
    print(f"\nTotal paramètres à apprendre : {total:,}")
    print(f"Mémoire estimée du modèle    : ~{total * 4 / 1e6:.1f} MB")
    print("=" * 55)


def entrainer_modele(model, X_train, y_train, X_val, y_val, class_weight=None):
    #                                                         

    print("\n" + "=" * 55)
    print("ENTRAÎNEMENT")
    print("=" * 55)
    print(f"Images train  : {X_train.shape[0]}")
    print(f"Images val    : {X_val.shape[0]}")
    print(f"Epochs (max)  : {MAX_EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Augmentation  : {'ON' if USE_AUGMENTATION else 'OFF'}")
    print("Early stop    : monitor=val_loss, patience=4")
    print("LR scheduler  : ReduceLROnPlateau")
    print()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    historique = model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,   
        verbose=1
    )

    return historique

def afficher_courbes(historique):
    """
    Affiche les courbes d'accuracy et de loss.

    GUIDE DE DIAGNOSTIC :
    ─────────────────────────────────────────────────────────
    OVERFITTING   → train monte fort, val stagne ou descend
    UNDERFITTING  → les deux restent bas (<70%)
    BON APPRENTISSAGE → les deux montent ensemble, écart <10%
    ─────────────────────────────────────────────────────────
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(historique.history['accuracy']) + 1)

    axes[0].plot(epochs, historique.history['accuracy'],
                 label='Train', linewidth=2, color='#378ADD')
    axes[0].plot(epochs, historique.history['val_accuracy'],
                 label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
    axes[0].set_title('Accuracy par epoch', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, historique.history['loss'],
                 label='Train', linewidth=2, color='#378ADD')
    axes[1].plot(epochs, historique.history['val_loss'],
                 label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
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
    Recharger avec : model = load_model('models/cnn_baseline.keras')
    """
    import os
    os.makedirs('models', exist_ok=True)
    model.save(chemin)
    print(f"\nModèle sauvegardé : {chemin}")


def evaluer_modele(model, X_test, y_test, y_test_raw):
    from sklearn.metrics import classification_report

    print("\n" + "=" * 55)
    print("ÉVALUATION FINALE — données jamais vues")
    print("=" * 55)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Loss finale     : {loss:.4f}")
    print(f"  Accuracy finale : {accuracy:.1%}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    noms_classes = ['notumor', 'meningioma', 'pituitary', 'glioma']
    print("\n" + classification_report(
        y_test_raw, y_pred, target_names=noms_classes
    ))
    print("=" * 55)
    return accuracy