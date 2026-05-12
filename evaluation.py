"""
evaluation.py — Phase de test finale

Ce fichier contient TOUTES les métriques demandées par le CdC :
  - Accuracy finale sur X_test
  - Matrice de confusion
  - Sensibilité (Recall) par classe  → cible > 95%
  - F2-Score                         → privilégie le recall
  - ROC-AUC par classe               → discrimination par stade

À appeler UNE SEULE FOIS après l'entraînement complet.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    fbeta_score
)
from tensorflow.keras.models import load_model
from src.preprocessing import obtenir_donnees

# =============================================================
# CONFIGURATION
# =============================================================

NOMS_CLASSES  = ['notumor', 'meningioma', 'pituitary', 'glioma']
CHEMIN_MODELE = 'models/cnn_baseline.keras'
CHEMIN_TRAIN = 'data/Training'
CHEMIN_TEST = 'data/Testing'
SEUIL_CONFIANCE = 0.7   # en dessous → "révision manuelle" (CdC)

COULEURS = {
    'notumor':    '#639922',
    'meningioma': '#378ADD',
    'pituitary':  '#EF9F27',
    'glioma':     '#E24B4A'
}


# =============================================================
# FONCTION 1 : Charger modèle + faire les prédictions
#
# model.predict() retourne pour chaque image un vecteur
# de 4 probabilités. Ex : [0.03, 0.08, 0.07, 0.82]
#
# np.argmax() prend le maximum → classe prédite = 3 (glioma)
# =============================================================

def preparer_predictions(model, X_test, y_test, y_test_raw):
    """
    Calcule toutes les prédictions une seule fois
    et retourne tout ce dont les autres fonctions ont besoin.
    """
    print("Calcul des prédictions sur X_test...")

    # y_proba : tableau (N, 4) — les 4 probabilités pour chaque image
    # C'est la sortie brute du Softmax
    y_proba = model.predict(X_test, verbose=0)

    # y_pred : tableau (N,) — la classe prédite (0, 1, 2 ou 3)
    # On prend simplement le maximum de chaque vecteur
    y_pred = np.argmax(y_proba, axis=1)

    # Confiance maximale pour chaque prédiction
    # Ex : [0.03, 0.08, 0.07, 0.82] → confiance = 0.82
    confiances = np.max(y_proba, axis=1)

    print(f"  {len(y_pred)} images évaluées")
    print(f"  Confiance moyenne : {confiances.mean():.1%}")
    print(f"  Prédictions < {SEUIL_CONFIANCE:.0%} (révision manuelle) : "
          f"{(confiances < SEUIL_CONFIANCE).sum()} images "
          f"({(confiances < SEUIL_CONFIANCE).mean():.1%})")

    return y_proba, y_pred, confiances


# =============================================================
# FONCTION 2 : Score global
# =============================================================

def afficher_score_global(model, X_test, y_test):
    """
    Évalue loss et accuracy sur X_test.
    C'est le score honnête — X_test n'a jamais été vu pendant l'entraînement.
    """
    print("\n" + "=" * 55)
    print("SCORE FINAL SUR X_TEST")
    print("=" * 55)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {accuracy:.1%}")

    if accuracy >= 0.90:
        print("  ✅ Excellent — baseline validée")
    elif accuracy >= 0.80:
        print("  ✅ Bon résultat pour un CNN from scratch")
    else:
        print("  ⚠️  En dessous des attentes")

    return accuracy


# =============================================================
# FONCTION 3 : Matrice de confusion
#
# La matrice de confusion montre pour chaque vraie classe
# combien d'images ont été classées dans chaque classe prédite.
#
# Ligne = vraie classe
# Colonne = classe prédite
#
# La diagonale = prédictions correctes
# Hors diagonale = erreurs
#
# Exemple à lire :
#   Ligne glioma, colonne meningioma = 12
#   → 12 gliomas ont été confondus avec des meningiomas
#   C'est dangereux en médecine (faux négatif sur tumeur maligne)
# =============================================================

def afficher_matrice_confusion(y_test_raw, y_pred, dossier_resultats="."):
    """
    Affiche et sauvegarde la matrice de confusion.
    """
    print("\n" + "=" * 55)
    print("MATRICE DE CONFUSION")
    print("=" * 55)

    cm = confusion_matrix(y_test_raw, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalisation : on affiche les pourcentages par ligne
    # (par rapport au vrai label) pour mieux comparer
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Afficher les valeurs dans chaque cellule
    for i in range(len(NOMS_CLASSES)):
        for j in range(len(NOMS_CLASSES)):
            valeur = cm[i, j]
            pct    = cm_norm[i, j]
            # Texte blanc sur fond foncé, noir sur fond clair
            couleur_texte = 'white' if pct > 0.5 else 'black'
            ax.text(j, i, f"{valeur}\n({pct:.0%})",
                    ha='center', va='center',
                    fontsize=11, color=couleur_texte, fontweight='bold')

    ax.set_xticks(range(len(NOMS_CLASSES)))
    ax.set_yticks(range(len(NOMS_CLASSES)))
    ax.set_xticklabels(NOMS_CLASSES, rotation=30, ha='right')
    ax.set_yticklabels(NOMS_CLASSES)
    ax.set_xlabel('Classe prédite', fontsize=12)
    ax.set_ylabel('Vraie classe', fontsize=12)
    ax.set_title('Matrice de confusion (% par ligne)', fontsize=13)

    plt.tight_layout()
    chemin_save = os.path.join(dossier_resultats, 'matrice_confusion.png')
    plt.savefig(chemin_save, dpi=100)
    plt.show()
    print(f"  Sauvegardée : {chemin_save}")

    # Analyse des erreurs critiques
    print("\n  Erreurs critiques (confusion entre classes tumorales) :")
    for i in range(1, len(NOMS_CLASSES)):      # vraies classes tumorales
        for j in range(len(NOMS_CLASSES)):     # classes prédites
            if i != j and cm[i, j] > 0:
                print(f"    {NOMS_CLASSES[i]} → prédit {NOMS_CLASSES[j]} : "
                      f"{cm[i, j]} cas ({cm_norm[i, j]:.1%})")

    return cm


# =============================================================
# FONCTION 4 : Sensibilité (Recall) et F2-Score
#
# SENSIBILITÉ = parmi tous les vrais malades,
#               combien le modèle a-t-il détectés ?
#
# Ex : 100 vrais gliomas → modèle en détecte 92 → sensibilité = 92%
#      Les 8 manquants = FAUX NÉGATIFS → patient non diagnostiqué
#
# En oncologie, un faux négatif est bien pire qu'un faux positif :
#   Faux positif → examen complémentaire inutile (coût, stress)
#   Faux négatif → tumeur non traitée → risque vital
#
# F2-Score : comme le F1, mais le recall compte DOUBLE
#   F1 = harmonic mean(precision, recall)  — équilibré
#   F2 = recall² × 2 / (precision + recall×2) — privilégie recall
# =============================================================

def afficher_sensibilite_f2(y_test_raw, y_pred):
    """
    Affiche sensibilité et F2-score par classe.
    Compare à la cible du CdC (> 95%).
    """
    print("\n" + "=" * 55)
    print("SENSIBILITÉ (RECALL) ET F2-SCORE")
    print("=" * 55)
    print(f"  Cible CdC : sensibilité > 95% sur chaque classe\n")

    rapport = classification_report(
        y_test_raw, y_pred,
        target_names=NOMS_CLASSES,
        output_dict=True
    )

    # F2-score global (moyenne pondérée)
    f2_global = fbeta_score(
        y_test_raw, y_pred,
        beta=2,              # beta=2 → recall compte 2x plus que precision
        average='weighted'
    )

    print(f"  {'Classe':<12} {'Recall':>8} {'Précision':>10} "
          f"{'F2-score':>10} {'Statut':>8}")
    print("  " + "-" * 52)

    for classe in NOMS_CLASSES:
        recall    = rapport[classe]['recall']
        precision = rapport[classe]['precision']
        support   = rapport[classe]['support']

        # F2 par classe
        f2_classe = fbeta_score(
            y_test_raw, y_pred,
            beta=2,
            average=None,
            labels=[NOMS_CLASSES.index(classe)]
        )[0]

        statut = "✅" if recall >= 0.95 else "⚠️ "
        print(f"  {classe:<12} {recall:>7.1%} {precision:>9.1%} "
              f"{f2_classe:>9.1%} {statut:>8}  (n={support})")

    print("  " + "-" * 52)
    print(f"  {'F2 global':<12} {'':>8} {'':>10} {f2_global:>9.1%}")

    return rapport, f2_global


# =============================================================
# FONCTION 5 : ROC-AUC par classe
#
# ROC = Receiver Operating Characteristic
# AUC = Area Under the Curve
#
# Pour chaque classe, on trace la courbe qui montre comment
# le modèle distingue "c'est cette classe" vs "ce n'est pas cette classe"
# en faisant varier le seuil de décision.
#
# AUC = 1.0 → parfait
# AUC = 0.5 → aussi bon qu'un tirage aléatoire
# AUC > 0.95 → excellent pour un diagnostic médical
# =============================================================

def afficher_roc_auc(y_test_raw, y_proba, dossier_resultats="."):
    """
    Calcule et affiche les courbes ROC pour chaque classe.
    Utilise la stratégie One-vs-Rest : pour chaque classe,
    on calcule si le modèle distingue bien "cette classe" vs "toutes les autres".
    """
    print("\n" + "=" * 55)
    print("ROC-AUC PAR CLASSE")
    print("=" * 55)

    # One-hot encoding de y_test_raw pour sklearn
    # Ex : 3 → [0, 0, 0, 1]
    from tensorflow.keras.utils import to_categorical
    y_test_onehot = to_categorical(y_test_raw, num_classes=len(NOMS_CLASSES))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, classe in enumerate(NOMS_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, i], y_proba[:, i])
        auc = roc_auc_score(y_test_onehot[:, i], y_proba[:, i])

        statut = "✅" if auc >= 0.95 else "⚠️ "
        print(f"  {statut} {classe:<12} : AUC = {auc:.4f}")

        ax.plot(fpr, tpr,
                label=f"{classe} (AUC={auc:.3f})",
                color=list(COULEURS.values())[i],
                linewidth=2)

    # Ligne diagonale = modèle aléatoire (AUC = 0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC=0.5)')

    ax.set_xlabel('Taux de faux positifs', fontsize=12)
    ax.set_ylabel('Taux de vrais positifs (Sensibilité)', fontsize=12)
    ax.set_title('Courbes ROC par classe', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    chemin_save = os.path.join(dossier_resultats, 'roc_auc.png')
    plt.savefig(chemin_save, dpi=100)
    plt.show()
    print(f"  Sauvegardée : {chemin_save}")


# =============================================================
# FONCTION 6 : Distribution des confiances
#
# Le CdC demande : si la probabilité max < 0.7,
# déclencher une demande de révision manuelle.
#
# Ce graphique montre combien d'images sont dans ce cas.
# =============================================================

def afficher_distribution_confiances(y_test_raw, y_pred, confiances, dossier_resultats="."):
    """
    Montre la distribution des scores de confiance
    et identifie les cas qui nécessitent une révision humaine.
    """
    print("\n" + "=" * 55)
    print(f"DÉTECTION DES CAS INCERTAINS (seuil = {SEUIL_CONFIANCE:.0%})")
    print("=" * 55)

    cas_certains   = confiances >= SEUIL_CONFIANCE
    cas_incertains = confiances < SEUIL_CONFIANCE

    print(f"  Cas certains   (≥ {SEUIL_CONFIANCE:.0%}) : "
          f"{cas_certains.sum()} ({cas_certains.mean():.1%})")
    print(f"  Cas incertains (< {SEUIL_CONFIANCE:.0%}) : "
          f"{cas_incertains.sum()} ({cas_incertains.mean():.1%})"
          f" → révision manuelle recommandée")

    # Accuracy sur les cas certains seulement
    acc_certains = (y_pred[cas_certains] == y_test_raw[cas_certains]).mean()
    print(f"\n  Accuracy sur cas certains seulement : {acc_certains:.1%}")
    print(f"  (le modèle est plus fiable quand il est confiant)")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confiances[cas_certains],   bins=20, color='#378ADD',
            alpha=0.7, label=f'Certains (≥{SEUIL_CONFIANCE:.0%})')
    ax.hist(confiances[cas_incertains], bins=20, color='#E24B4A',
            alpha=0.7, label=f'Incertains (<{SEUIL_CONFIANCE:.0%})')
    ax.axvline(SEUIL_CONFIANCE, color='black', linestyle='--',
               linewidth=1.5, label=f'Seuil = {SEUIL_CONFIANCE:.0%}')
    ax.set_xlabel('Score de confiance (probabilité max)', fontsize=12)
    ax.set_ylabel('Nombre d\'images', fontsize=12)
    ax.set_title('Distribution des scores de confiance', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chemin_save = os.path.join(dossier_resultats, 'distribution_confiances.png')
    plt.savefig(chemin_save, dpi=100)
    plt.show()
    print(f"  Sauvegardée : {chemin_save}")


# =============================================================
# FONCTION PRINCIPALE : tout enchaîner
# =============================================================

def evaluer_complet(chemin_modele, X_test, y_test, y_test_raw):
    """
    Lance l'évaluation complète dans l'ordre logique.
    Appeler cette fonction depuis main.py.
    """
    nom_modele = os.path.basename(chemin_modele).replace('.keras', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier_resultats = os.path.join("results", f"{nom_modele}_{timestamp}")
    os.makedirs(dossier_resultats, exist_ok=True)

    print("\n" + "=" * 55)
    print("ÉVALUATION COMPLÈTE — PHASE DE TEST")
    print(f"Dossier de résultats : {dossier_resultats}")
    print("=" * 55)
    print("Chargement du modèle sauvegardé...")
    # On ajoute compile=False pour éviter l'erreur liée à l'optimizer Adam (Keras/TF versions issue)
    model = load_model(chemin_modele, compile=False)
    # Re-compilation manuelle nécessaire pour pouvoir utiliser model.evaluate()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 1. Prédictions
    y_proba, y_pred, confiances = preparer_predictions(
        model, X_test, y_test, y_test_raw
    )

    # 2. Score global
    afficher_score_global(model, X_test, y_test)

    # 3. Matrice de confusion
    afficher_matrice_confusion(y_test_raw, y_pred, dossier_resultats)

    # 4. Sensibilité + F2
    afficher_sensibilite_f2(y_test_raw, y_pred)

    # 5. ROC-AUC
    afficher_roc_auc(y_test_raw, y_proba, dossier_resultats)

    # 6. Confiances
    afficher_distribution_confiances(y_test_raw, y_pred, confiances, dossier_resultats)

    print("\n" + "=" * 55)
    print("ÉVALUATION TERMINÉE")
    print(f"Fichiers générés dans : {dossier_resultats}/")
    print("  - matrice_confusion.png")
    print("  - roc_auc.png")
    print("  - distribution_confiances.png")
    print("=" * 55)


def executer_evaluation_complete(X_test=None, y_test=None, y_test_raw=None,
                                 chemin_modele=CHEMIN_MODELE,
                                 chemin_train=CHEMIN_TRAIN,
                                 chemin_test=CHEMIN_TEST,
                                 forcer_recalcul=False):
    """
    Lance l'évaluation complète sans réentraîner le modèle.

    Si les données de test ne sont pas fournies, elles sont chargées
    automatiquement via le cache existant.
    """
    if X_test is None or y_test is None or y_test_raw is None:
        _, _, _, _, _, X_test, y_test, y_test_raw = obtenir_donnees(
            chemin_train,
            chemin_test,
            forcer_recalcul=forcer_recalcul
        )

    return evaluer_complet(chemin_modele, X_test, y_test, y_test_raw)


if __name__ == "__main__":
    executer_evaluation_complete()

