import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.preprocessing import obtenir_donnees
from src.preprocessing_tl import obtenir_donnees as obtenir_donnees_tl
import numpy as np
from src.model import (
    construire_modele,
    afficher_architecture,
    entrainer_modele,
    afficher_courbes,
    sauvegarder_modele
)
from src.TransferLearned import (
    construire_modele as construire_modele_tl,
    afficher_architecture as afficher_architecture_tl,
    entrainer_modele as entrainer_modele_tl,
    afficher_courbes as afficher_courbes_tl,
    sauvegarder_modele as sauvegarder_modele_tl,
    fine_tune_model as fine_tune_model_tl
)
from sklearn.utils.class_weight import compute_class_weight

from evaluation import executer_evaluation_complete

# =============================================================
# CHEMINS
# =============================================================

CHEMIN_TRAIN = os.path.join('data', 'Training')
CHEMIN_TEST  = os.path.join('data', 'Testing')

def lancer_entrainement():
    # ÉTAPE 1 : Charger les données
    X_train, y_train, y_train_raw, \
    X_val,   y_val,                \
    X_test,  y_test,  y_test_raw = obtenir_donnees(
        CHEMIN_TRAIN,
        CHEMIN_TEST,
        forcer_recalcul=False
    )

    # =========================================================
    # ✅ NOUVEAU : Class weights
    # On dit au modèle : "les erreurs sur meningioma et glioma
    # comptent plus que les autres"
    # =========================================================
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train_raw)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_raw)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    # Boost manuel des classes difficiles
    class_weight[1] = 2.5   # meningioma — le pire
    class_weight[3] = 1.8   # glioma — deuxième pire

    print(f"\nClass weights appliqués : {class_weight}")
    # Affichera : {0: 1.0, 1: 2.5, 2: 1.0, 3: 1.8}

    # ÉTAPE 2 : Construire le modèle
    model = construire_modele()
    afficher_architecture(model)

    # ÉTAPE 3 : Entraîner — on passe class_weight ici
    historique = entrainer_modele(model, X_train, y_train, X_val, y_val,
                                  class_weight=class_weight)  # ← ajout

    # ÉTAPE 4, 5 : inchangées
    afficher_courbes(historique)
    sauvegarder_modele(model)
    executer_evaluation_complete(X_test, y_test, y_test_raw)

def lancer_evaluation_seule():
    # =========================================================
    # Mode évaluation: on recharge les données + le modèle sauvegardé
    # sans réentraîner.
    # =========================================================
    _, _, _, \
    _, _, \
    X_test, y_test, y_test_raw = obtenir_donnees(
        CHEMIN_TRAIN,
        CHEMIN_TEST,
        forcer_recalcul=True
    )

    executer_evaluation_complete(X_test, y_test, y_test_raw)


def lancer_transfer_learning():
    # =========================================================
    # ÉTAPE 1 : Charger les données (avec cache automatique)
    # =========================================================
    X_train, y_train, y_train_raw, \
    X_val,   y_val,                \
    X_test,  y_test,  y_test_raw = obtenir_donnees_tl(
        CHEMIN_TRAIN,
        CHEMIN_TEST,
        forcer_recalcul=True
    )

    # =========================================================
    # ÉTAPE 2 : Construire le modèle Transfer Learning
    # =========================================================
    model = construire_modele_tl()
    afficher_architecture_tl(model)

    labels = y_train_raw
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    class_weight[1] = 2.5
    class_weight[3] = 1.8
    print(f"Class weights: {class_weight}")

    # =========================================================
    # ÉTAPE 3 : Entraîner
    # =========================================================
    historique = entrainer_modele_tl(model, X_train, y_train, X_val, y_val,
                                     class_weight=class_weight)

    # =========================================================
    # ÉTAPE 3b : Fine-tuning (dégeler dernières couches du backbone)
    # Calculer class_weight depuis y_train_raw pour favoriser classes difficiles
    # =========================================================
    labels = y_train_raw
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Class weight used for fine-tuning: {class_weight}")

    # Fine-tune: dégeler les dernières 30 couches, LR très faible
    ft_hist = fine_tune_model_tl(model, X_train, y_train, X_val, y_val,
                                 num_unfreeze=30, fine_tune_epochs=8,
                                 fine_lr=1e-5, class_weight=class_weight)

    if ft_hist is not None:
        afficher_courbes_tl(ft_hist)

    # =========================================================
    # ÉTAPE 4 : Visualiser les résultats
    # =========================================================
    afficher_courbes_tl(historique)

    # =========================================================
    # ÉTAPE 5 : Sauvegarder puis évaluer sur X_test
    # =========================================================
    chemin_tl = 'models/transfer_efficientnetb0.keras'
    sauvegarder_modele_tl(model, chemin=chemin_tl)
    executer_evaluation_complete(
        X_test,
        y_test,
        y_test_raw,
        chemin_modele=chemin_tl
    )


def main():
    print("\n==============================")
    print("MODE D'EXÉCUTION DU PROJET")
    print("==============================")
    print("1 - Entraîner le modèle")
    print("2 - Évaluer le modèle sauvegardé")
    print("3 - Entraîner en Transfer Learning")
    choix = input("Choisis un mode (1/2/3) : ").strip().lower()

    if choix in ("1", "train", "entraînement", "entrainement"):
        lancer_entrainement()
    elif choix in ("2", "eval", "évaluation", "evaluation"):
        lancer_evaluation_seule()
    elif choix in ("3", "tl", "transfer", "transferlearning", "transfer_learning"):
        lancer_transfer_learning()
    else:
        print("Choix invalide. Lance de nouveau et choisis 1, 2 ou 3.")


if __name__ == "__main__":
    main()