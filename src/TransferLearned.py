from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
	Input, Dense, Dropout, GlobalAveragePooling2D,
	RandomFlip, RandomRotation, RandomZoom, RandomContrast,
	Rescaling
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
 
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


IMG_SIZE = 160
NUM_CLASSES = 4
MAX_EPOCHS = 30
BATCH_SIZE = 32
USE_AUGMENTATION = True
MODEL_VERSION = "TL_EfficientNetB0_head_matched"


if USE_AUGMENTATION:
	DATA_AUGMENTATION = Sequential([
		RandomRotation(0.08),
		RandomZoom(0.10),
		RandomFlip("horizontal"),
		RandomContrast(0.1),
	], name="data_augmentation")
else:
	DATA_AUGMENTATION = None


def construire_modele():
	"""
	Transfer learning en gardant le head identique au modèle from-scratch
	(GAP → Dropout(0.5) → Dense(128, l2) → Dropout(0.4) → softmax)
	Le backbone est EfficientNetB0 (ImageNet) gelé.
	"""
	model = Sequential(name=f"Brain_Tumor_{MODEL_VERSION}")
	model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

	if DATA_AUGMENTATION is not None:
		model.add(DATA_AUGMENTATION)

	# preprocessing TL: preprocessing_tl.py fournit des images en [0,1]
	# On multiplie par 255 via une couche Rescaling (sérialisable).
	model.add(Rescaling(255.0, name='rescale_255'))

	base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
	base_model.trainable = False

	model.add(base_model)

	# Head identique au from-scratch
	model.add(GlobalAveragePooling2D())
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
	model.add(Dropout(0.4))
	model.add(Dense(NUM_CLASSES, activation='softmax'))

	model.compile(
		optimizer=Adam(learning_rate=0.0005),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	return model


def afficher_architecture(model):
	print('\n' + '=' * 55)
	print('ARCHITECTURE TRANSFER LEARNING (matched head)')
	print('=' * 55)
	model.summary()
	total = model.count_params()
	print(f"\nTotal paramètres : {total:,}")
	print(f"Mémoire estimée  : ~{total * 4 / 1e6:.1f} MB")
	print('=' * 55)


def entrainer_modele(model, X_train, y_train, X_val, y_val):
	print('\n' + '=' * 55)
	print('ENTRAÎNEMENT TRANSFER LEARNING')
	print('=' * 55)
	print(f"Images train  : {X_train.shape[0]}")
	print(f"Images val    : {X_val.shape[0]}")
	print(f"Epochs (max)  : {MAX_EPOCHS}")
	print(f"Batch size    : {BATCH_SIZE}")
	print(f"Augmentation   : {'ON' if USE_AUGMENTATION else 'OFF'}")
	print('Backbone      : EfficientNetB0 (frozen)')

	callbacks = [
		EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
		ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
	]

	historique = model.fit(
		X_train, y_train,
		epochs=MAX_EPOCHS,
		batch_size=BATCH_SIZE,
		validation_data=(X_val, y_val),
		callbacks=callbacks,
		verbose=1
	)

	return historique


def fine_tune_model(model, X_train, y_train, X_val, y_val,
					num_unfreeze=30, fine_tune_epochs=10, fine_lr=1e-5,
					class_weight=None, backbone_name='efficientnetb0'):
	"""
	Fine-tuning: dégèle les `num_unfreeze` dernières couches du backbone
	et réentraîne à un learning rate très faible.

	- `model` doit contenir le backbone EfficientNetB0 comme une sous-couche.
	- `num_unfreeze`: nombre de couches à rendre entraînables (depuis la fin).
	- `class_weight`: dict optionnel passé à model.fit
	"""
	# Trouver explicitement le backbone EfficientNetB0.
	# Important: ne pas prendre la couche d'augmentation (Sequential) par erreur.
	backbone = None
	for layer in model.layers:
		layer_name = getattr(layer, 'name', '').lower()
		if backbone_name in layer_name:
			backbone = layer
			break

	if backbone is None:
		for layer in model.layers:
			if layer.__class__.__name__.lower() in ('functional', 'model') and hasattr(layer, 'layers'):
				if 'efficientnet' in getattr(layer, 'name', '').lower():
					backbone = layer
					break

	if backbone is None:
		print('Backbone non trouvé — impossible de fine-tuner.')
		return None

	# Comptage des sous-couches
	total_layers = len(backbone.layers)
	n = min(num_unfreeze, total_layers)

	# Geler d'abord tout, puis dégeler les dernières n couches
	for l in backbone.layers:
		l.trainable = False

	for l in backbone.layers[-n:]:
		l.trainable = True

	print(f"Fine-tuning : dégèlement des dernières {n} couches du backbone (sur {total_layers}).")

	# Recompiler avec LR faible
	model.compile(
		optimizer=Adam(learning_rate=fine_lr),
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	callbacks = [
		EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
		ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
	]

	historique = model.fit(
		X_train, y_train,
		epochs=fine_tune_epochs,
		batch_size=BATCH_SIZE,
		validation_data=(X_val, y_val),
		callbacks=callbacks,
		class_weight=class_weight,
		verbose=1
	)

	return historique


def afficher_courbes(historique):
	fig, axes = plt.subplots(1, 2, figsize=(13, 4))
	epochs = range(1, len(historique.history['accuracy']) + 1)

	axes[0].plot(epochs, historique.history['accuracy'], label='Train', linewidth=2, color='#378ADD')
	axes[0].plot(epochs, historique.history['val_accuracy'], label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
	axes[0].set_title('Accuracy par epoch (TL)', fontsize=12)
	axes[0].set_xlabel('Epoch')
	axes[0].set_ylabel('Accuracy')
	axes[0].legend()
	axes[0].set_ylim(0, 1)
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(epochs, historique.history['loss'], label='Train', linewidth=2, color='#378ADD')
	axes[1].plot(epochs, historique.history['val_loss'], label='Validation', linewidth=2, color='#E24B4A', linestyle='--')
	axes[1].set_title('Loss par epoch (TL)', fontsize=12)
	axes[1].set_xlabel('Epoch')
	axes[1].set_ylabel('Loss')
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('courbes_entrainement_tl.png', dpi=100)
	plt.show()
	print('Courbes sauvegardées : courbes_entrainement_tl.png')


def sauvegarder_modele(model, chemin='models/transfer_efficientnetb0.keras'):
	import os
	os.makedirs('models', exist_ok=True)
	model.save(chemin)
	print(f"\nModèle sauvegardé : {chemin}")
	print('Pour le recharger plus tard :')
	print("  from tensorflow.keras.models import load_model")
	print(f"  model = load_model('{chemin}')")

