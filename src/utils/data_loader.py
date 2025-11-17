"""
Module de chargement et sauvegarde des données.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


def load_data(
    data_dir: Union[str, Path] = '../data brut',
    train_x_file: str = 'X_train_update.csv',
    train_y_file: str = 'Y_train.csv',
    test_x_file: str = 'X_test_update.csv'
) -> tuple:
    """
    Charge les données d'entraînement et de test.
    
    Parameters
    ----------
    data_dir : str ou Path
        Dossier contenant les fichiers de données
    train_x_file : str
        Nom du fichier d'entraînement X
    train_y_file : str
        Nom du fichier d'entraînement Y
    test_x_file : str
        Nom du fichier de test X
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test)
    """
    data_dir = Path(data_dir)
    
    print("🔄 Chargement des données...")
    
    # Charger les données
    X_train = pd.read_csv(data_dir / train_x_file)
    y_train = pd.read_csv(data_dir / train_y_file)
    X_test = pd.read_csv(data_dir / test_x_file)
    
    print(f"✅ Données chargées avec succès !")
    print(f"  - X_train : {X_train.shape[0]:,} lignes × {X_train.shape[1]} colonnes")
    print(f"  - y_train : {y_train.shape[0]:,} lignes × {y_train.shape[1]} colonnes")
    print(f"  - X_test  : {X_test.shape[0]:,} lignes × {X_test.shape[1]} colonnes")
    
    return X_train, y_train, X_test


def save_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False
) -> None:
    """
    Sauvegarde un DataFrame dans un fichier CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à sauvegarder
    filepath : str ou Path
        Chemin du fichier de sortie
    index : bool, default=False
        Sauvegarder l'index
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=index)
    print(f"✅ Données sauvegardées : {filepath}")

