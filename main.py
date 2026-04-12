import os
from src.preprocessing import obtenir_donnees
from src.model import (
    construire_modele,
    afficher_architecture,
    entrainer_modele,
    afficher_courbes,
    sauvegarder_modele
)

# =============================================================
# CHEMINS
# =============================================================

CHEMIN_TRAIN = os.path.join('data', 'Training')
CHEMIN_TEST  = os.path.join('data', 'Testing')

# =============================================================
# ÉTAPE 1 : Charger les données (avec cache automatique)
# =============================================================

X_train, y_train, y_train_raw, \
X_test,  y_test,  y_test_raw = obtenir_donnees(
    CHEMIN_TRAIN,
    CHEMIN_TEST,
    forcer_recalcul=False
)

# =============================================================
# ÉTAPE 2 : Construire le modèle
# =============================================================

model = construire_modele()
afficher_architecture(model)

# =============================================================
# ÉTAPE 3 : Entraîner
# =============================================================

historique = entrainer_modele(model, X_train, y_train, X_test, y_test)

# =============================================================
# ÉTAPE 4 : Visualiser les résultats
# =============================================================

afficher_courbes(historique)

# =============================================================
# ÉTAPE 5 : Sauvegarder le modèle entraîné
# =============================================================

sauvegarder_modele(model)