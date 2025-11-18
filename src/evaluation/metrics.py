"""
Module d'évaluation avec métriques adaptées au déséquilibre de classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from typing import Dict, Any, Optional
from collections import Counter


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Évalue un modèle avec plusieurs métriques.
    
    Parameters
    ----------
    y_true : np.ndarray
        Labels réels
    y_pred : np.ndarray
        Labels prédits
    average : str
        Type de moyenne pour les métriques ('macro', 'micro', 'weighted')
        
    Returns
    -------
    dict
        Dictionnaire des métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> None:
    """
    Affiche un rapport de classification détaillé.
    
    Parameters
    ----------
    y_true : np.ndarray
        Labels réels
    y_pred : np.ndarray
        Labels prédits
    target_names : list, optional
        Noms des classes
    """
    print("="*80)
    print("RAPPORT DE CLASSIFICATION")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Métriques globales
    metrics = evaluate_model(y_true, y_pred)
    print("\n📊 Métriques Globales :")
    print(f"  - Accuracy : {metrics['accuracy']:.4f}")
    print(f"  - F1-score (macro) : {metrics['f1_macro']:.4f}")
    print(f"  - F1-score (weighted) : {metrics['f1_weighted']:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    figsize: tuple = (15, 12),
    normalize: bool = True
) -> None:
    """
    Affiche la matrice de confusion.
    
    Parameters
    ----------
    y_true : np.ndarray
        Labels réels
    y_pred : np.ndarray
        Labels prédits
    class_names : list, optional
        Noms des classes
    figsize : tuple
        Taille de la figure
    normalize : bool
        Normaliser la matrice (pourcentages)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matrice de Confusion (Normalisée)'
    else:
        fmt = 'd'
        title = 'Matrice de Confusion'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Nombre'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Vraies valeurs', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None
) -> None:
    """
    Compare la distribution des classes réelles et prédites.
    
    Parameters
    ----------
    y_true : np.ndarray
        Labels réels
    y_pred : np.ndarray
        Labels prédits
    class_names : list, optional
        Noms des classes
    """
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    all_classes = sorted(set(y_true) | set(y_pred))
    
    true_values = [true_counts.get(c, 0) for c in all_classes]
    pred_values = [pred_counts.get(c, 0) for c in all_classes]
    
    x = np.arange(len(all_classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width/2, true_values, width, label='Réel', alpha=0.8)
    ax.bar(x + width/2, pred_values, width, label='Prédit', alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Nombre d\'occurrences', fontsize=12)
    ax.set_title('Distribution des Classes : Réel vs Prédit', fontsize=14)
    ax.set_xticks(x)
    if class_names:
        ax.set_xticklabels([class_names[i] if i < len(class_names) else str(c) for i, c in enumerate(all_classes)], rotation=45, ha='right')
    else:
        ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

