"""
Module de réseaux de neurones pour la classification de texte.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
    # Maximiser l'utilisation CPU (threads intra/inter op)
    import os
    n_cores = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(n_cores)
    tf.config.threading.set_inter_op_parallelism_threads(n_cores)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow n'est pas installé. Installez-le avec: pip install tensorflow")


def create_mlp_model(
    input_dim: int,
    num_classes: int,
    hidden_layers: list = [128, 64],
    dropout_rate: float = 0.3,
    activation: str = 'relu'
) -> Any:
    """
    Crée un modèle MLP (Multi-Layer Perceptron) pour la classification.
    
    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée
    num_classes : int
        Nombre de classes
    hidden_layers : list
        Liste des tailles des couches cachées
    dropout_rate : float
        Taux de dropout
    activation : str
        Fonction d'activation
        
    Returns
    -------
    keras.Model
        Modèle MLP compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé")
    
    model = models.Sequential()
    
    # Couche d'entrée
    model.add(layers.Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
    model.add(layers.Dropout(dropout_rate))
    
    # Couches cachées
    for layer_size in hidden_layers[1:]:
        model.add(layers.Dense(layer_size, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_cnn_model(
    input_dim: int,
    num_classes: int,
    filters: list = [64, 128],
    kernel_sizes: list = [3, 3],
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Any:
    """
    Crée un modèle CNN 1D pour la classification de texte.
    
    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée
    num_classes : int
        Nombre de classes
    filters : list
        Nombre de filtres par couche
    kernel_sizes : list
        Tailles des kernels
    dense_units : int
        Nombre d'unités dans la couche dense
    dropout_rate : float
        Taux de dropout
        
    Returns
    -------
    keras.Model
        Modèle CNN compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé")
    
    # Reshape pour CNN 1D
    model = models.Sequential()
    model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
    
    # Couches de convolution
    for i, (filters_count, kernel_size) in enumerate(zip(filters, kernel_sizes)):
        model.add(layers.Conv1D(filters_count, kernel_size, activation='relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout_rate))
    
    # Flatten
    model.add(layers.Flatten())
    
    # Couche dense
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(
    input_dim: int,
    num_classes: int,
    lstm_units: list = [64, 32],
    dense_units: int = 64,
    dropout_rate: float = 0.3
) -> Any:
    """
    Crée un modèle LSTM pour la classification de texte.
    
    Parameters
    ----------
    input_dim : int
        Dimension des features d'entrée
    num_classes : int
        Nombre de classes
    lstm_units : list
        Nombre d'unités LSTM par couche
    dense_units : int
        Nombre d'unités dans la couche dense
    dropout_rate : float
        Taux de dropout
        
    Returns
    -------
    keras.Model
        Modèle LSTM compilé
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé")
    
    # Reshape pour LSTM (nécessite une séquence)
    model = models.Sequential()
    model.add(layers.Reshape((input_dim, 1), input_shape=(input_dim,)))
    
    # Couches LSTM
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout_rate))
    
    # Couche dense
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_neural_network(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 1,
    early_stopping: bool = True,
    patience: int = 5
) -> Dict[str, Any]:
    """
    Entraîne un réseau de neurones.
    
    Parameters
    ----------
    model : keras.Model
        Modèle à entraîner
    X_train : np.ndarray
        Features d'entraînement
    y_train : np.ndarray
        Labels d'entraînement
    X_val : np.ndarray, optional
        Features de validation
    y_val : np.ndarray, optional
        Labels de validation
    epochs : int
        Nombre d'époques
    batch_size : int
        Taille des batches
    verbose : int
        Niveau de verbosité
    early_stopping : bool
        Utiliser early stopping
    patience : int
        Patience pour early stopping
        
    Returns
    -------
    dict
        Historique d'entraînement et modèle entraîné
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow n'est pas installé")
    
    callbacks_list = []
    
    if early_stopping and X_val is not None:
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks_list.append(early_stop)
    
    validation_data = (X_val, y_val) if X_val is not None else None
    
    print(f"🔄 Entraînement du modèle neural...")
    import time
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=verbose
    )
    elapsed = time.time() - start_time
    print(f"Temps d'entraînement : {elapsed} secondes")
    print(f"✅ Modèle neural entraîné avec succès")
    
    return {
        'model': model,
        'history': history.history
    }

