import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
NUM_CLASSES = 4
MAX_EPOCHS = 25
BATCH_SIZE = 32
USE_AUGMENTATION = True

if USE_AUGMENTATION:
    DATA_AUGMENTATION = Sequential([
        RandomRotation(0.1),
        RandomZoom(0.15),
        RandomFlip("horizontal"),
        RandomContrast(0.15),
    ], name="data_augmentation")
else:
    DATA_AUGMENTATION = None


def construire_modele():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    model = Sequential(name="Transfer_Learning_EfficientNetB0")
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    if DATA_AUGMENTATION is not None:
        model.add(DATA_AUGMENTATION)

    model.add(base_model)

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def afficher_architecture(model):
    print("\n" + "=" * 55)
    print("ARCHITECTURE TRANSFER LEARNING")
    print("=" * 55)
    model.summary()

    total = model.count_params()
    trainable = sum([w.shape.num_elements() for w in model.trainable_weights])
    print(f"\nTotal paramètres     : {total:,}")
    print(f"Paramètres entraînables: {trainable:,}")
    print(f"Paramètres gelés     : {total - trainable:,}")
    print("=" * 55)


def entrainer_modele(model, X_train, y_train, X_val, y_val, class_weight=None):
    X_train_processed = preprocess_input(X_train)
    X_val_processed = preprocess_input(X_val)

    print("\n" + "=" * 55)
    print("ENTRAÎNEMENT TRANSFER LEARNING (backbone gelé)")
    print("=" * 55)
    print(f"Images train  : {X_train.shape[0]}")
    print(f"Images val    : {X_val.shape[0]}")
    print(f"Epochs (max)  : {MAX_EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print("Backbone      : EfficientNetB0 (gelé)")
    if class_weight:
        print(f"Class weight  : {class_weight}")
    print()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
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
        X_train_processed, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_processed, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    return historique


def fine_tune_model(model, X_train, y_train, X_val, y_val,
                    num_unfreeze=50, fine_tune_epochs=10,
                    fine_lr=1e-4, class_weight=None):
    X_train_processed = preprocess_input(X_train)
    X_val_processed = preprocess_input(X_val)

    print("\n" + "=" * 55)
    print(f"FINE-TUNING (dégel {num_unfreeze} dernières couches)")
    print("=" * 55)

    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'name') and 'efficientnet' in layer.name.lower():
            base_model = layer
            break
        if hasattr(layer, 'layers') and not isinstance(layer, Sequential):
            base_model = layer
            break

    if base_model is not None:
        print(f"Base model found: {base_model.name}")
        base_model.trainable = True
        
        unfreeze_from = len(base_model.layers) - num_unfreeze
        print(f"Total layers: {len(base_model.layers)}, unfreezing from index {unfreeze_from}")
        
        for i, layer in enumerate(base_model.layers):
            if i >= unfreeze_from:
                layer.trainable = True
            else:
                layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"Couches dégelées : {trainable_count} / {len(base_model.layers)}")

    model.compile(
        optimizer=Adam(learning_rate=fine_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

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
            min_lr=1e-7,
            verbose=1
        )
    ]

    historique = model.fit(
        X_train_processed, y_train,
        epochs=fine_tune_epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_processed, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    return historique


def afficher_courbes(historique, finetune_hist=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    if finetune_hist is not None:
        acc = historique.history['accuracy'] + finetune_hist.history['accuracy']
        val_acc = historique.history['val_accuracy'] + finetune_hist.history['val_accuracy']
        loss = historique.history['loss'] + finetune_hist.history['loss']
        val_loss = historique.history['val_loss'] + finetune_hist.history['val_loss']
    else:
        acc = historique.history['accuracy']
        val_acc = historique.history['val_accuracy']
        loss = historique.history['loss']
        val_loss = historique.history['val_loss']

    epochs = range(1, len(acc) + 1)

    axes[0].plot(epochs, acc, label='Train', linewidth=2, color='#378ADD')
    axes[0].plot(epochs, val_acc, label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
    axes[0].set_title('Accuracy par epoch', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    if finetune_hist is not None:
        axes[0].axvline(x=len(historique.history['accuracy']), color='green', linestyle=':', label='Fine-tune start')

    axes[1].plot(epochs, loss, label='Train', linewidth=2, color='#378ADD')
    axes[1].plot(epochs, val_loss, label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
    axes[1].set_title('Loss par epoch', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('courbes_transfer_learning.png', dpi=100)
    plt.show()
    print("Courbes sauvegardées : courbes_transfer_learning.png")


def sauvegarder_modele(model, chemin='models/transfer_learning.keras'):
    os.makedirs('models', exist_ok=True)
    model.save(chemin)
    print(f"\nModèle sauvegardé : {chemin}")