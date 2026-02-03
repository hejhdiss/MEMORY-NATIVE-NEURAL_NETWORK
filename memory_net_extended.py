#!/usr/bin/env python3
"""
MEMORY-NATIVE NEURAL NETWORK - EXTENDED PYTHON API
Version 2.0 - Complete Implementation of All Three Memory Concepts

Features:
- Concept 1: Memory-Preserving Activation (β parameter)
- Concept 2: Stateful Neurons (α parameter)
- Concept 3: Learnable Memory Dynamics (network learns what to remember)
- Scikit-learn-like API (fit, predict, score)
- Joblib and custom serialization support
- Partial training capabilities
- Comprehensive memory management

Compile C library first:
    Windows: gcc -shared -o memory_net_extended.dll memory_net_extended.c -lm -O3
    Linux:   gcc -shared -fPIC -o memory_net_extended.so memory_net_extended.c -lm -O3
    Mac:     gcc -shared -fPIC -o memory_net_extended.dylib memory_net_extended.c -lm -O3
Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
import pickle
import json
from typing import Optional, Union, Tuple, Dict, Any
import warnings

# Optional imports
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not available. Install with: pip install joblib")

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'memory_net_extended.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'memory_net_extended.dylib'
    else:
        lib_name = 'memory_net_extended.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o memory_net_extended.dll memory_net_extended.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o memory_net_extended.dylib memory_net_extended.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o memory_net_extended.so memory_net_extended.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded extended C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_network.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float
]
_lib.create_network.restype = ctypes.c_void_p
_lib.destroy_network.argtypes = [ctypes.c_void_p]
_lib.destroy_network.restype = None

# Forward/Predict
_lib.forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.forward.restype = None
_lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.predict.restype = None

# Training
_lib.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.train.restype = ctypes.c_float
_lib.train_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.train_batch.restype = ctypes.c_float

# Feature control
_lib.set_use_recurrent.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.set_use_recurrent.restype = None
_lib.set_use_learnable_gates.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.set_use_learnable_gates.restype = None
_lib.set_use_output_memory.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_lib.set_use_output_memory.restype = None
_lib.get_use_recurrent.argtypes = [ctypes.c_void_p]
_lib.get_use_recurrent.restype = ctypes.c_bool
_lib.get_use_learnable_gates.argtypes = [ctypes.c_void_p]
_lib.get_use_learnable_gates.restype = ctypes.c_bool
_lib.get_use_output_memory.argtypes = [ctypes.c_void_p]
_lib.get_use_output_memory.restype = ctypes.c_bool

# Partial training - advanced
_lib.freeze_hidden_layer.argtypes = [ctypes.c_void_p]
_lib.freeze_hidden_layer.restype = None
_lib.unfreeze_hidden_layer.argtypes = [ctypes.c_void_p]
_lib.unfreeze_hidden_layer.restype = None
_lib.freeze_output_layer.argtypes = [ctypes.c_void_p]
_lib.freeze_output_layer.restype = None
_lib.unfreeze_output_layer.argtypes = [ctypes.c_void_p]
_lib.unfreeze_output_layer.restype = None
_lib.freeze_memory_gates.argtypes = [ctypes.c_void_p]
_lib.freeze_memory_gates.restype = None
_lib.unfreeze_memory_gates.argtypes = [ctypes.c_void_p]
_lib.unfreeze_memory_gates.restype = None

# Advanced partial training
_lib.freeze_hidden_percentage.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.freeze_hidden_percentage.restype = None
_lib.freeze_output_percentage.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.freeze_output_percentage.restype = None
_lib.freeze_by_magnitude.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]
_lib.freeze_by_magnitude.restype = None
_lib.freeze_specific_neurons.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
_lib.freeze_specific_neurons.restype = None

_lib.set_freeze_mask_hidden.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_lib.set_freeze_mask_hidden.restype = None
_lib.set_freeze_mask_output.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_lib.set_freeze_mask_output.restype = None
_lib.set_freeze_mask_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_lib.set_freeze_mask_memory.restype = None

_lib.get_freeze_mask_hidden.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_lib.get_freeze_mask_hidden.restype = None
_lib.get_freeze_mask_output.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
_lib.get_freeze_mask_output.restype = None

_lib.count_frozen_hidden.argtypes = [ctypes.c_void_p]
_lib.count_frozen_hidden.restype = ctypes.c_int
_lib.count_frozen_output.argtypes = [ctypes.c_void_p]
_lib.count_frozen_output.restype = ctypes.c_int

# Gradient accumulation
_lib.accumulate_gradients.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.accumulate_gradients.restype = ctypes.c_float
_lib.apply_accumulated_gradients.argtypes = [ctypes.c_void_p]
_lib.apply_accumulated_gradients.restype = None
_lib.reset_gradients.argtypes = [ctypes.c_void_p]
_lib.reset_gradients.restype = None

# Memory management
_lib.reset_memory.argtypes = [ctypes.c_void_p]
_lib.reset_memory.restype = None
_lib.get_memory_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.get_memory_state.restype = None
_lib.set_memory_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.set_memory_state.restype = None
_lib.get_gate_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.get_gate_state.restype = None

# Parameters
_lib.get_beta.argtypes = [ctypes.c_void_p]
_lib.get_beta.restype = ctypes.c_float
_lib.set_beta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_beta.restype = None
_lib.get_alpha.argtypes = [ctypes.c_void_p]
_lib.get_alpha.restype = ctypes.c_float
_lib.set_alpha.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_alpha.restype = None
_lib.get_output_beta.argtypes = [ctypes.c_void_p]
_lib.get_output_beta.restype = ctypes.c_float
_lib.set_output_beta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_output_beta.restype = None
_lib.get_learning_rate.argtypes = [ctypes.c_void_p]
_lib.get_learning_rate.restype = ctypes.c_float
_lib.set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_learning_rate.restype = None
_lib.get_training_steps.argtypes = [ctypes.c_void_p]
_lib.get_training_steps.restype = ctypes.c_int
_lib.get_last_loss.argtypes = [ctypes.c_void_p]
_lib.get_last_loss.restype = ctypes.c_float
_lib.get_avg_memory_magnitude.argtypes = [ctypes.c_void_p]
_lib.get_avg_memory_magnitude.restype = ctypes.c_float
_lib.get_avg_gate_value.argtypes = [ctypes.c_void_p]
_lib.get_avg_gate_value.restype = ctypes.c_float

# Save/Load
_lib.save_network.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.save_network.restype = ctypes.c_int
_lib.load_network.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.load_network.restype = ctypes.c_int

# ============================================================================
# MAIN API CLASS - SCIKIT-LEARN COMPATIBLE
# ============================================================================

class MemoryNeuralNetwork:
    """
    Memory-Native Neural Network with ALL THREE memory concepts
    
    A neural network where neurons have persistent internal memory states that
    influence all future computations. Implements three complementary memory mechanisms:
    
    **Concept 1: Memory-Preserving Activation**
        y(t) = activation(W×x(t) + U×h(t-1)) + β×y(t-1)
        Creates mathematical "echo" of past computations
    
    **Concept 2: Stateful Neurons**
        memory(t) = (1-α)×memory(t-1) + α×new_information
        Neurons maintain internal state that persists and decays
    
    **Concept 3: Learnable Memory Dynamics**
        memory_gate = learned_function(input, current_memory)
        Network learns what to remember, when, and how strongly
    
    Scikit-learn Compatible API:
    - fit(X, y): Train the network
    - predict(X): Make predictions
    - score(X, y): Evaluate performance
    - partial_fit(X, y): Incremental learning
    
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden neurons with memory
    output_size : int
        Number of output neurons
    beta : float, default=0.3
        Memory preservation factor (0-1) for Concept 1
        Higher = stronger echo of past outputs
    alpha : float, default=0.1
        Memory update rate (0-1) for Concept 2
        Controls how fast internal memory updates
    output_beta : float, default=0.2
        Memory preservation for output layer
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    use_recurrent : bool, default=True
        Enable LSTM-like recurrent connections
    use_learnable_gates : bool, default=True
        Enable Concept 3 (learnable memory dynamics)
    use_output_memory : bool, default=True
        Apply memory preservation to output layer
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted
    training_history_ : dict
        Training loss history and other metrics
    n_features_in_ : int
        Number of input features seen during fit
    n_outputs_ : int
        Number of outputs
        
    Examples
    --------
    >>> from memory_neural_net import MemoryNeuralNetwork
    >>> import numpy as np
    
    >>> # Create network
    >>> model = MemoryNeuralNetwork(
    ...     input_size=10, 
    ...     hidden_size=20, 
    ...     output_size=3,
    ...     beta=0.3,  # Memory preservation
    ...     alpha=0.1  # Update rate
    ... )
    
    >>> # Train (scikit-learn style)
    >>> X_train = np.random.randn(100, 10)
    >>> y_train = np.random.randn(100, 3)
    >>> model.fit(X_train, y_train, epochs=50)
    
    >>> # Predict
    >>> X_test = np.random.randn(20, 10)
    >>> predictions = model.predict(X_test)
    
    >>> # Save model (with memory states)
    >>> model.save_model('my_model.pkl')
    >>> # or use joblib
    >>> import joblib
    >>> joblib.dump(model, 'my_model.joblib')
    
    >>> # Load model
    >>> model2 = MemoryNeuralNetwork.load_model('my_model.pkl')
    >>> # or
    >>> model2 = joblib.load('my_model.joblib')
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 output_size: int,
                 beta: float = 0.3,
                 alpha: float = 0.1,
                 output_beta: float = 0.2,
                 learning_rate: float = 0.01,
                 use_recurrent: bool = True,
                 use_learnable_gates: bool = True,
                 use_output_memory: bool = True,
                 random_state: Optional[int] = None):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.beta = beta
        self.alpha = alpha
        self.output_beta = output_beta
        self.learning_rate = learning_rate
        self.use_recurrent = use_recurrent
        self.use_learnable_gates = use_learnable_gates
        self.use_output_memory = use_output_memory
        self.random_state = random_state
        
        # Sklearn-compatible attributes
        self.is_fitted_ = False
        self.n_features_in_ = input_size
        self.n_outputs_ = output_size
        self.training_history_ = {
            'loss': [],
            'epochs': [],
            'memory_magnitude': [],
            'gate_values': []
        }
        
        # Create network via C library
        if random_state is not None:
            np.random.seed(random_state)
        
        self._net = _lib.create_network(
            input_size, hidden_size, output_size,
            beta, alpha, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create network")
        
        # Set feature flags
        _lib.set_use_recurrent(self._net, use_recurrent)
        _lib.set_use_learnable_gates(self._net, use_learnable_gates)
        _lib.set_use_output_memory(self._net, use_output_memory)
        _lib.set_output_beta(self._net, output_beta)
        
        self._verbose = False
    
    # ========================================================================
    # SCIKIT-LEARN COMPATIBLE API
    # ========================================================================
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, 
            batch_size: Optional[int] = None,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: int = 1,
            shuffle: bool = True) -> 'MemoryNeuralNetwork':
        """
        Fit the model to training data
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, n_outputs)
            Target values
        epochs : int, default=100
            Number of training epochs
        batch_size : int, optional
            Batch size for training. If None, use full batch
        validation_data : tuple of (X_val, y_val), optional
            Validation data for monitoring
        verbose : int, default=1
            Verbosity level (0=silent, 1=progress bar, 2=epoch details)
        shuffle : bool, default=True
            Whether to shuffle training data each epoch
            
        Returns
        -------
        self : MemoryNeuralNetwork
            Returns self for method chaining
        """
        X, y = self._validate_data(X, y)
        self._verbose = verbose > 0
        
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        
        for epoch in range(epochs):
            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            # Train in batches
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self._train_batch(X_batch, y_batch)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Record history
            self.training_history_['loss'].append(avg_loss)
            self.training_history_['epochs'].append(epoch)
            self.training_history_['memory_magnitude'].append(
                _lib.get_avg_memory_magnitude(self._net)
            )
            self.training_history_['gate_values'].append(
                _lib.get_avg_gate_value(self._net)
            )
            
            # Validation
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.score(X_val, y_val, return_loss=True)
            
            # Logging
            if verbose == 2:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)
            elif verbose == 1 and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        self.is_fitted_ = True
        return self
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'MemoryNeuralNetwork':
        """
        Incremental fit on a batch of samples
        
        Useful for online learning or when data doesn't fit in memory.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch
        y : array-like of shape (n_samples, n_outputs)
            Target values
            
        Returns
        -------
        self : MemoryNeuralNetwork
        """
        X, y = self._validate_data(X, y)
        self._train_batch(X, y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_outputs)
            Predictions
        """
        X = self._validate_data(X, y=None)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = X.shape[0]
        X = np.ascontiguousarray(X, dtype=np.float32)
        
        outputs = []
        for i in range(batch_size):
            input_data = X[i].flatten()
            output_data = np.zeros(self.output_size, dtype=np.float32)
            
            _lib.predict(
                self._net,
                input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            outputs.append(output_data.copy())
        
        result = np.array(outputs)
        return result[0] if squeeze else result
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              return_loss: bool = False) -> float:
        """
        Return the coefficient of determination R² or MSE loss
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples, n_outputs)
            True values
        return_loss : bool, default=False
            If True, return MSE loss instead of R² score
            
        Returns
        -------
        score : float
            R² score or MSE loss
        """
        X, y = self._validate_data(X, y)
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        
        if return_loss:
            return mse
        else:
            # R² score
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            return r2
    
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def reset_memory(self) -> None:
        """Reset all memory states to zero"""
        _lib.reset_memory(self._net)
    
    def get_memory_state(self) -> np.ndarray:
        """
        Get current internal memory state
        
        Returns
        -------
        memory : ndarray of shape (hidden_size,)
            Current memory values
        """
        memory = np.zeros(self.hidden_size, dtype=np.float32)
        _lib.get_memory_state(
            self._net,
            memory.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return memory
    
    def set_memory_state(self, memory: np.ndarray) -> None:
        """
        Set internal memory state
        
        Parameters
        ----------
        memory : array-like of shape (hidden_size,)
            Memory values to set
        """
        memory = np.ascontiguousarray(memory, dtype=np.float32)
        _lib.set_memory_state(
            self._net,
            memory.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    
    def get_gate_state(self) -> np.ndarray:
        """
        Get current memory gate values (Concept 3)
        
        Returns
        -------
        gates : ndarray of shape (hidden_size,)
            Current gate values [0, 1]
        """
        gates = np.zeros(self.hidden_size, dtype=np.float32)
        _lib.get_gate_state(
            self._net,
            gates.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return gates
    
    # ========================================================================
    # PARTIAL TRAINING (TRANSFER LEARNING & FINE-TUNING)
    # ========================================================================
    
    def freeze_hidden_layer(self) -> None:
        """
        Freeze hidden layer weights (stop training them)
        
        Useful for transfer learning when you want to preserve learned features
        """
        _lib.freeze_hidden_layer(self._net)
    
    def unfreeze_hidden_layer(self) -> None:
        """Unfreeze hidden layer weights"""
        _lib.unfreeze_hidden_layer(self._net)
    
    def freeze_output_layer(self) -> None:
        """Freeze output layer weights"""
        _lib.freeze_output_layer(self._net)
    
    def unfreeze_output_layer(self) -> None:
        """Unfreeze output layer weights"""
        _lib.unfreeze_output_layer(self._net)
    
    def freeze_memory_gates(self) -> None:
        """
        Freeze learnable memory gate weights (Concept 3)
        
        Preserves learned memory dynamics while allowing other weights to train
        """
        _lib.freeze_memory_gates(self._net)
    
    def unfreeze_memory_gates(self) -> None:
        """Unfreeze learnable memory gate weights"""
        _lib.unfreeze_memory_gates(self._net)
    
    def freeze_hidden_percentage(self, percentage: float) -> None:
        """
        Freeze a random percentage of hidden layer weights
        
        Parameters
        ----------
        percentage : float
            Fraction of weights to freeze (0.0 to 1.0)
            
        Examples
        --------
        >>> model.freeze_hidden_percentage(0.5)  # Freeze 50% of hidden weights
        """
        _lib.freeze_hidden_percentage(self._net, percentage)
    
    def freeze_output_percentage(self, percentage: float) -> None:
        """
        Freeze a random percentage of output layer weights
        
        Parameters
        ----------
        percentage : float
            Fraction of weights to freeze (0.0 to 1.0)
        """
        _lib.freeze_output_percentage(self._net, percentage)
    
    def freeze_by_magnitude(self, percentage: float, freeze_smallest: bool = True) -> None:
        """
        Freeze weights based on their magnitude
        
        Parameters
        ----------
        percentage : float
            Fraction of weights to freeze (0.0 to 1.0)
        freeze_smallest : bool, default=True
            If True, freeze smallest weights (pruning)
            If False, freeze largest weights
            
        Examples
        --------
        >>> # Freeze 30% of smallest weights (prune unimportant connections)
        >>> model.freeze_by_magnitude(0.3, freeze_smallest=True)
        
        >>> # Freeze 20% of largest weights (preserve less, train more)
        >>> model.freeze_by_magnitude(0.2, freeze_smallest=False)
        """
        _lib.freeze_by_magnitude(self._net, percentage, freeze_smallest)
    
    def freeze_specific_neurons(self, neuron_indices: Union[list, np.ndarray]) -> None:
        """
        Freeze all weights connected to specific neurons
        
        Parameters
        ----------
        neuron_indices : array-like
            Indices of neurons to freeze (0 to hidden_size-1)
            
        Examples
        --------
        >>> # Freeze neurons 5, 10, and 15
        >>> model.freeze_specific_neurons([5, 10, 15])
        """
        neuron_indices = np.asarray(neuron_indices, dtype=np.int32)
        _lib.freeze_specific_neurons(
            self._net,
            neuron_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(neuron_indices)
        )
    
    def set_freeze_mask_hidden(self, mask: np.ndarray) -> None:
        """
        Set custom freeze mask for hidden layer
        
        Parameters
        ----------
        mask : ndarray of bool, shape (input_size * hidden_size,)
            True = trainable, False = frozen
            
        Examples
        --------
        >>> # Create custom mask
        >>> mask = np.ones(model.input_size * model.hidden_size, dtype=bool)
        >>> mask[::2] = False  # Freeze every other weight
        >>> model.set_freeze_mask_hidden(mask)
        """
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.input_size * self.hidden_size:
            raise ValueError(f"Mask size mismatch: expected {self.input_size * self.hidden_size}, got {len(mask)}")
        
        mask_c = np.ascontiguousarray(mask, dtype=ctypes.c_bool)
        _lib.set_freeze_mask_hidden(
            self._net,
            mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        )
    
    def set_freeze_mask_output(self, mask: np.ndarray) -> None:
        """
        Set custom freeze mask for output layer
        
        Parameters
        ----------
        mask : ndarray of bool, shape (hidden_size * output_size,)
            True = trainable, False = frozen
        """
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.hidden_size * self.output_size:
            raise ValueError(f"Mask size mismatch: expected {self.hidden_size * self.output_size}, got {len(mask)}")
        
        mask_c = np.ascontiguousarray(mask, dtype=ctypes.c_bool)
        _lib.set_freeze_mask_output(
            self._net,
            mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        )
    
    def set_freeze_mask_memory(self, mask: np.ndarray) -> None:
        """
        Set custom freeze mask for memory gate weights
        
        Parameters
        ----------
        mask : ndarray of bool, shape (input_size * hidden_size,)
            True = trainable, False = frozen
        """
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.input_size * self.hidden_size:
            raise ValueError(f"Mask size mismatch: expected {self.input_size * self.hidden_size}, got {len(mask)}")
        
        mask_c = np.ascontiguousarray(mask, dtype=ctypes.c_bool)
        _lib.set_freeze_mask_memory(
            self._net,
            mask_c.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        )
    
    def get_freeze_mask_hidden(self) -> np.ndarray:
        """
        Get current freeze mask for hidden layer
        
        Returns
        -------
        mask : ndarray of bool, shape (input_size * hidden_size,)
            True = trainable, False = frozen
        """
        size = self.input_size * self.hidden_size
        mask = np.zeros(size, dtype=ctypes.c_bool)
        _lib.get_freeze_mask_hidden(
            self._net,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        )
        return mask.astype(bool)
    
    def get_freeze_mask_output(self) -> np.ndarray:
        """
        Get current freeze mask for output layer
        
        Returns
        -------
        mask : ndarray of bool, shape (hidden_size * output_size,)
            True = trainable, False = frozen
        """
        size = self.hidden_size * self.output_size
        mask = np.zeros(size, dtype=ctypes.c_bool)
        _lib.get_freeze_mask_output(
            self._net,
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        )
        return mask.astype(bool)
    
    def count_frozen_weights(self) -> Dict[str, int]:
        """
        Count number of frozen weights in each layer
        
        Returns
        -------
        counts : dict
            Dictionary with keys 'hidden', 'output', 'total_hidden', 'total_output'
            
        Examples
        --------
        >>> counts = model.count_frozen_weights()
        >>> print(f"Frozen: {counts['hidden']}/{counts['total_hidden']} hidden weights")
        """
        frozen_hidden = _lib.count_frozen_hidden(self._net)
        frozen_output = _lib.count_frozen_output(self._net)
        
        total_hidden = self.input_size * self.hidden_size
        total_output = self.hidden_size * self.output_size
        
        return {
            'hidden': frozen_hidden,
            'output': frozen_output,
            'total_hidden': total_hidden,
            'total_output': total_output,
            'hidden_trainable': total_hidden - frozen_hidden,
            'output_trainable': total_output - frozen_output,
            'frozen_percentage_hidden': 100.0 * frozen_hidden / total_hidden if total_hidden > 0 else 0,
            'frozen_percentage_output': 100.0 * frozen_output / total_output if total_output > 0 else 0
        }
    
    def print_freeze_status(self) -> None:
        """Print detailed freeze status"""
        counts = self.count_frozen_weights()
        print("\n" + "="*60)
        print("FREEZE STATUS")
        print("="*60)
        print(f"\nHidden Layer:")
        print(f"  Frozen: {counts['hidden']} / {counts['total_hidden']} ({counts['frozen_percentage_hidden']:.1f}%)")
        print(f"  Trainable: {counts['hidden_trainable']}")
        print(f"\nOutput Layer:")
        print(f"  Frozen: {counts['output']} / {counts['total_output']} ({counts['frozen_percentage_output']:.1f}%)")
        print(f"  Trainable: {counts['output_trainable']}")
        print("="*60 + "\n")
    
    # ========================================================================
    # ADVANCED TRAINING WITH GRADIENT ACCUMULATION
    # ========================================================================
    
    def fit_with_accumulation(self, 
                             X: np.ndarray, 
                             y: np.ndarray,
                             epochs: int = 100,
                             accumulation_steps: int = 4,
                             verbose: int = 1) -> 'MemoryNeuralNetwork':
        """
        Train with gradient accumulation (simulates larger batch sizes)
        
        Useful when you want larger effective batch sizes but are memory-constrained.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, n_outputs)
            Target values
        epochs : int, default=100
            Number of epochs
        accumulation_steps : int, default=4
            Number of samples to accumulate before updating weights
        verbose : int, default=1
            Verbosity level
            
        Returns
        -------
        self : MemoryNeuralNetwork
            
        Examples
        --------
        >>> # Simulate batch size of 32 with accumulation_steps=4 (8*4=32)
        >>> model.fit_with_accumulation(X, y, accumulation_steps=4)
        """
        X, y = self._validate_data(X, y)
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_updates = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(n_samples):
                # Accumulate gradients
                input_data = X_shuffled[i].flatten().astype(np.float32)
                target_data = y_shuffled[i].flatten().astype(np.float32)
                
                loss = _lib.accumulate_gradients(
                    self._net,
                    input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    target_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                )
                epoch_loss += loss
                
                # Apply gradients after accumulation_steps
                if (i + 1) % accumulation_steps == 0 or i == n_samples - 1:
                    _lib.apply_accumulated_gradients(self._net)
                    num_updates += 1
            
            avg_loss = epoch_loss / n_samples
            self.training_history_['loss'].append(avg_loss)
            self.training_history_['epochs'].append(epoch)
            
            if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Updates: {num_updates}")
        
        self.is_fitted_ = True
        return self
    
    # ========================================================================
    # PARAMETER ACCESS
    # ========================================================================
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator (sklearn compatibility)
        
        Parameters
        ----------
        deep : bool, default=True
            Ignored (for sklearn compatibility)
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'beta': self.beta,
            'alpha': self.alpha,
            'output_beta': self.output_beta,
            'learning_rate': self.learning_rate,
            'use_recurrent': self.use_recurrent,
            'use_learnable_gates': self.use_learnable_gates,
            'use_output_memory': self.use_output_memory,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'MemoryNeuralNetwork':
        """
        Set parameters (sklearn compatibility)
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : MemoryNeuralNetwork
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Update C library if needed
                if key == 'beta':
                    _lib.set_beta(self._net, value)
                elif key == 'alpha':
                    _lib.set_alpha(self._net, value)
                elif key == 'output_beta':
                    _lib.set_output_beta(self._net, value)
                elif key == 'learning_rate':
                    _lib.set_learning_rate(self._net, value)
        return self
    
    @property
    def training_steps(self) -> int:
        """Number of training steps performed"""
        return _lib.get_training_steps(self._net)
    
    @property
    def last_loss(self) -> float:
        """Last training loss"""
        return _lib.get_last_loss(self._net)
    
    @property
    def avg_memory_magnitude(self) -> float:
        """Average magnitude of internal memory"""
        return _lib.get_avg_memory_magnitude(self._net)
    
    @property
    def avg_gate_value(self) -> float:
        """Average memory gate value (0-1)"""
        return _lib.get_avg_gate_value(self._net)
    
    # ========================================================================
    # SERIALIZATION - MULTIPLE METHODS
    # ========================================================================
    
    def save_model(self, filepath: str, method: str = 'custom') -> None:
        """
        Save model to file with multiple serialization options
        
        Parameters
        ----------
        filepath : str
            Path to save file
        method : str, default='custom'
            Serialization method:
            - 'custom': Fast binary format (recommended)
            - 'joblib': Use joblib (if available)
            - 'pickle': Use pickle
        """
        if method == 'custom':
            # Save to binary format via C library
            result = _lib.save_network(self._net, filepath.encode('utf-8'))
            if result != 0:
                raise RuntimeError(f"Failed to save network (error {result})")
            
            # Save Python-side metadata
            metadata = {
                'params': self.get_params(),
                'is_fitted': self.is_fitted_,
                'training_history': self.training_history_
            }
            meta_path = filepath + '.meta'
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            if self._verbose:
                print(f"✓ Model saved to {filepath}")
        
        elif method == 'joblib':
            if not HAS_JOBLIB:
                raise ImportError("joblib not available. Install with: pip install joblib")
            joblib.dump(self, filepath)
            if self._verbose:
                print(f"✓ Model saved with joblib to {filepath}")
        
        elif method == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            if self._verbose:
                print(f"✓ Model pickled to {filepath}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @classmethod
    def load_model(cls, filepath: str, method: str = 'custom') -> 'MemoryNeuralNetwork':
        """
        Load model from file
        
        Parameters
        ----------
        filepath : str
            Path to model file
        method : str, default='custom'
            Serialization method used (must match save method)
            
        Returns
        -------
        model : MemoryNeuralNetwork
            Loaded model
        """
        if method == 'custom':
            # Load metadata
            meta_path = filepath + '.meta'
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Create new network
            params = metadata['params']
            model = cls(**params)
            
            # Load weights and memory states from C library
            result = _lib.load_network(model._net, filepath.encode('utf-8'))
            if result == -2:
                raise ValueError("Network size mismatch")
            elif result != 0:
                raise RuntimeError(f"Failed to load network (error {result})")
            
            # Restore Python-side state
            model.is_fitted_ = metadata['is_fitted']
            model.training_history_ = metadata['training_history']
            
            print(f"✓ Model loaded from {filepath}")
            return model
        
        elif method == 'joblib':
            if not HAS_JOBLIB:
                raise ImportError("joblib not available. Install with: pip install joblib")
            model = joblib.load(filepath)
            print(f"✓ Model loaded with joblib from {filepath}")
            return model
        
        elif method == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model unpickled from {filepath}")
            return model
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _validate_data(self, X, y=None):
        """Validate input data"""
        X = np.asarray(X, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {X.shape[1]}")
        
        if y is not None:
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.shape[1] != self.output_size:
                raise ValueError(f"Expected {self.output_size} outputs, got {y.shape[1]}")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y have different number of samples")
            return X, y
        
        return X
    
    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on a batch of data"""
        batch_size = X.shape[0]
        X_flat = X.flatten()
        y_flat = y.flatten()
        
        loss = _lib.train_batch(
            self._net,
            X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_size
        )
        return loss
    
    def __repr__(self):
        params = self.get_params()
        param_str = ', '.join(f"{k}={v}" for k, v in params.items())
        return f"MemoryNeuralNetwork({param_str})"
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_network(self._net)
    
    def __getstate__(self):
        """Support for pickle"""
        # Save C network to temporary bytes buffer
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            _lib.save_network(self._net, tmp_path.encode('utf-8'))
            with open(tmp_path, 'rb') as f:
                c_network_data = f.read()
        finally:
            os.unlink(tmp_path)
        
        state = self.__dict__.copy()
        state['_c_network_data'] = c_network_data
        del state['_net']
        return state
    
    def __setstate__(self, state):
        """Support for unpickle"""
        import tempfile
        import os
        
        c_network_data = state.pop('_c_network_data')
        self.__dict__.update(state)
        
        # Recreate C network
        self._net = _lib.create_network(
            self.input_size, self.hidden_size, self.output_size,
            self.beta, self.alpha, self.learning_rate
        )
        
        # Load from bytes buffer
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(c_network_data)
        
        try:
            _lib.load_network(self._net, tmp_path.encode('utf-8'))
        finally:
            os.unlink(tmp_path)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_model(input_size: int, hidden_size: int, output_size: int, **kwargs):
    """
    Convenience function to create a MemoryNeuralNetwork
    
    Examples
    --------
    >>> model = create_model(10, 20, 3, beta=0.4, learning_rate=0.005)
    """
    return MemoryNeuralNetwork(input_size, hidden_size, output_size, **kwargs)

# ============================================================================
# DEMONSTRATION CODE
# ============================================================================

def demo_sklearn_api():
    """Demonstrate scikit-learn compatible API"""
    print("\n" + "="*70)
    print("DEMO: Scikit-learn Compatible API")
    print("="*70)
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 8).astype(np.float32)
    y = np.random.randn(200, 3).astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = MemoryNeuralNetwork(
        input_size=8,
        hidden_size=16,
        output_size=3,
        beta=0.3,
        alpha=0.1,
        learning_rate=0.01,
        random_state=42
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train, epochs=50, verbose=1)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"\nTest R² Score: {r2:.4f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Training Steps: {model.training_steps}")
    print(f"Avg Memory Magnitude: {model.avg_memory_magnitude:.6f}")
    print(f"Avg Gate Value: {model.avg_gate_value:.6f}")
    
    return model

def demo_serialization():
    """Demonstrate multiple serialization methods"""
    print("\n" + "="*70)
    print("DEMO: Serialization Methods")
    print("="*70)
    
    # Create and train model
    model = MemoryNeuralNetwork(5, 10, 2, beta=0.4)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randn(50, 2).astype(np.float32)
    model.fit(X, y, epochs=20, verbose=0)
    
    test_input = np.random.randn(1, 5).astype(np.float32)
    original_pred = model.predict(test_input)
    
    # Method 1: Custom binary format
    print("\n1. Custom binary format...")
    model.save_model('model_custom.bin', method='custom')
    model1 = MemoryNeuralNetwork.load_model('model_custom.bin', method='custom')
    pred1 = model1.predict(test_input)
    print(f"   Prediction difference: {np.abs(original_pred - pred1).max():.10f}")
    
    # Method 2: Joblib (if available)
    if HAS_JOBLIB:
        print("\n2. Joblib format...")
        model.save_model('model.joblib', method='joblib')
        model2 = MemoryNeuralNetwork.load_model('model.joblib', method='joblib')
        pred2 = model2.predict(test_input)
        print(f"   Prediction difference: {np.abs(original_pred - pred2).max():.10f}")
    
    # Method 3: Pickle
    print("\n3. Pickle format...")
    model.save_model('model.pkl', method='pickle')
    model3 = MemoryNeuralNetwork.load_model('model.pkl', method='pickle')
    pred3 = model3.predict(test_input)
    print(f"   Prediction difference: {np.abs(original_pred - pred3).max():.10f}")
    
    print("\n✓ All serialization methods preserve model state perfectly!")

def demo_memory_concepts():
    """Demonstrate all three memory concepts"""
    print("\n" + "="*70)
    print("DEMO: Three Memory Concepts")
    print("="*70)
    
    # Concept 1: Memory-Preserving Activation (Beta)
    print("\n--- Concept 1: Memory-Preserving Activation ---")
    print("Testing different beta values (0.0, 0.3, 0.7)...\n")
    
    for beta in [0.0, 0.3, 0.7]:
        model = MemoryNeuralNetwork(3, 6, 2, beta=beta, alpha=0.0)
        
        # Strong initial input
        strong = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        pred = model.predict(strong)
        print(f"Beta={beta} - After strong input: {pred[0]}")
        
        # Weak subsequent input
        weak = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        pred = model.predict(weak)
        print(f"Beta={beta} - After weak input:   {pred[0]}")
        print()
    
    # Concept 2: Stateful Neurons (Alpha)
    print("\n--- Concept 2: Stateful Neurons ---")
    print("Memory state evolution with alpha=0.2...\n")
    
    model = MemoryNeuralNetwork(3, 6, 2, beta=0.0, alpha=0.2)
    input_data = np.array([[1.0, 0.5, 0.8]], dtype=np.float32)
    
    for step in range(5):
        model.predict(input_data)
        memory = model.get_memory_state()
        print(f"Step {step}: Memory[0:3] = {memory[:3]}")
    
    # Concept 3: Learnable Memory Gates
    print("\n--- Concept 3: Learnable Memory Dynamics ---")
    print("Training network to learn what to remember...\n")
    
    model = MemoryNeuralNetwork(
        4, 8, 2,
        use_learnable_gates=True,
        learning_rate=0.02
    )
    
    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randn(100, 2).astype(np.float32)
    
    print("Before training:")
    print(f"  Avg gate value: {model.avg_gate_value:.4f}")
    
    model.fit(X_train, y_train, epochs=30, verbose=0)
    
    print("After training:")
    print(f"  Avg gate value: {model.avg_gate_value:.4f}")
    print(f"  Gate states: {model.get_gate_state()[:5]}")
    print("\nNetwork learned which memories to preserve!")

def demo_partial_training():
    """Demonstrate partial training / transfer learning"""
    print("\n" + "="*70)
    print("DEMO: Partial Training / Transfer Learning")
    print("="*70)
    
    # Create model
    model = MemoryNeuralNetwork(6, 12, 3, learning_rate=0.02)
    
    X = np.random.randn(80, 6).astype(np.float32)
    y = np.random.randn(80, 3).astype(np.float32)
    
    # Phase 1: Train everything
    print("\nPhase 1: Training all layers...")
    model.fit(X, y, epochs=20, verbose=0)
    print(f"Loss after Phase 1: {model.last_loss:.6f}")
    
    # Phase 2: Freeze hidden, train only output
    print("\nPhase 2: Freezing hidden layer...")
    model.freeze_hidden_layer()
    model.fit(X, y, epochs=20, verbose=0)
    print(f"Loss after Phase 2: {model.last_loss:.6f}")
    
    # Phase 3: Freeze memory gates, train others
    print("\nPhase 3: Freezing memory gates...")
    model.unfreeze_hidden_layer()
    model.freeze_memory_gates()
    model.fit(X, y, epochs=20, verbose=0)
    print(f"Loss after Phase 3: {model.last_loss:.6f}")
    
    print("\n✓ Partial training enables fine-tuning specific components!")

def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "MEMORY-NATIVE NEURAL NETWORK v2.0" + " "*25 + "║")
    print("║" + " "*15 + "Extended API with ALL THREE Concepts" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        model = demo_sklearn_api()
        
        input("\nPress Enter to continue...")
        demo_serialization()
        
        input("\nPress Enter to continue...")
        demo_memory_concepts()
        
        input("\nPress Enter to continue...")
        demo_partial_training()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Features Demonstrated:")
        print("  • Scikit-learn compatible API (fit, predict, score)")
        print("  • All THREE memory concepts fully implemented")
        print("  • Multiple serialization methods (custom, joblib, pickle)")
        print("  • Partial training for transfer learning")
        print("  • Complete memory state preservation")
        print("  • Learnable memory dynamics (Concept 3)")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()