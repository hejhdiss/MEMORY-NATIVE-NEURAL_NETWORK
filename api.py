#!/usr/bin/env python3
"""
Memory-Native Neural Networks API
==================================

A clean, unified API for two memory-native neural network architectures:
1. AMRC (Adaptive Memory Recurrent Cell) - Basic memory preservation
2. PMRC (Persistent Memory Recurrent Cell) - Advanced learnable memory dynamics

This API provides a simple interface to both models with:
- Scikit-learn compatible methods (fit, predict, score)
- Memory state management
- Partial training capabilities
- Model persistence
- Comprehensive configuration options
Licensed under GPL V3.
Author: @hejhdiss
Version: 1.0.0
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, Literal
import warnings
from pathlib import Path

# Import both underlying implementations
try:
    from memory_net_python import MemoryNativeNetwork as AMRC_Base
    HAS_AMRC = True
except ImportError:
    HAS_AMRC = False
    warnings.warn("AMRC base implementation not available. Compile memory_net_dll.c first.")

try:
    from memory_net_extended import MemoryNeuralNetwork as PMRC_Base
    HAS_PMRC = True
except ImportError:
    HAS_PMRC = False
    warnings.warn("PMRC base implementation not available. Compile memory_net_extended.c first.")


__version__ = "1.0.0"
__all__ = [
    'AMRC',
    'PMRC',
    'MemoryCell',
    'create_model',
]


# ============================================================================
# UNIFIED API CLASSES
# ============================================================================

class MemoryCell:
    """
    Base class for memory-native neural network cells.
    
    This class provides a unified interface for both AMRC and PMRC architectures,
    with common methods and properties for memory management, training, and inference.
    """
    
    def __init__(self, model_type: str, **kwargs):
        """Initialize a memory cell (do not call directly, use AMRC or PMRC)"""
        self.model_type = model_type
        self._model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, 
            batch_size: Optional[int] = None,
            validation_split: float = 0.0,
            verbose: int = 0,
            **kwargs) -> 'MemoryCell':
        """
        Train the model on data.
        
        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_samples, input_size)
        y : np.ndarray
            Training target data of shape (n_samples, output_size)
        epochs : int, default=100
            Number of training epochs
        batch_size : int, optional
            Batch size for training. If None, uses full batch
        validation_split : float, default=0.0
            Fraction of data to use for validation (0.0 to 1.0)
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress bar, 2=detailed)
        
        Returns
        -------
        self : MemoryCell
            Returns self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, input_size) or (input_size,)
        
        Returns
        -------
        predictions : np.ndarray
            Predicted outputs of shape (n_samples, output_size) or (output_size,)
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Test input data
        y : np.ndarray
            True target values
        
        Returns
        -------
        score : float
            R² coefficient of determination
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def reset_memory(self):
        """Reset internal memory state to zero."""
        raise NotImplementedError("Subclasses must implement reset_memory()")
    
    def get_memory_state(self) -> np.ndarray:
        """
        Get current internal memory state.
        
        Returns
        -------
        memory : np.ndarray
            Current memory state vector
        """
        raise NotImplementedError("Subclasses must implement get_memory_state()")
    
    def set_memory_state(self, memory: np.ndarray):
        """
        Set internal memory state.
        
        Parameters
        ----------
        memory : np.ndarray
            Memory state to set
        """
        raise NotImplementedError("Subclasses must implement set_memory_state()")
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        raise NotImplementedError("Subclasses must implement load()")


class AMRC(MemoryCell):
    """
    Adaptive Memory Recurrent Cell (AMRC)
    
    A memory-native neural network with basic memory preservation capabilities.
    Features include:
    - Memory-preserving activation (beta parameter)
    - Stateful neurons (alpha parameter)
    - Partial training (layer freezing)
    - High-performance C backend
    
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden neurons with memory
    output_size : int
        Number of output neurons
    beta : float, default=0.3
        Memory preservation factor (0.0 to 1.0)
        Higher values = stronger memory of past outputs
    alpha : float, default=0.1
        Memory update rate (0.0 to 1.0)
        Controls how fast internal memory updates
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    
    Examples
    --------
    >>> # Create and train a simple AMRC model
    >>> model = AMRC(input_size=10, hidden_size=20, output_size=5)
    >>> model.fit(X_train, y_train, epochs=50, verbose=1)
    >>> predictions = model.predict(X_test)
    
    >>> # Use memory state
    >>> memory = model.get_memory_state()
    >>> model.reset_memory()  # Clear memory
    >>> model.set_memory_state(memory)  # Restore memory
    
    >>> # Partial training
    >>> model.freeze_hidden_layer()
    >>> model.fit(X_new, y_new, epochs=20)  # Only trains output layer
    >>> model.unfreeze_hidden_layer()
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 beta: float = 0.3,
                 alpha: float = 0.1,
                 learning_rate: float = 0.01):
        
        if not HAS_AMRC:
            raise RuntimeError(
                "AMRC implementation not available. "
                "Please compile memory_net_dll.c first."
            )
        
        super().__init__("AMRC")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create underlying model
        self._model = AMRC_Base(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            beta=beta,
            alpha=alpha,
            learning_rate=learning_rate
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, 
            batch_size: Optional[int] = None,
            validation_split: float = 0.0,
            verbose: int = 0,
            reset_memory: bool = False) -> 'AMRC':
        """Train the AMRC model."""
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if reset_memory:
            self._model.reset_memory()
        
        # Use the underlying model's training method
        if batch_size is not None:
            # Batch training
            losses = []
            n_samples = X.shape[0]
            for epoch in range(epochs):
                epoch_loss = 0.0
                for i in range(0, n_samples, batch_size):
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    loss = self._model.train_batch(X_batch, y_batch)
                    epoch_loss += loss
                losses.append(epoch_loss / (n_samples // batch_size))
                
                if verbose > 0 and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {losses[-1]:.6f}")
        else:
            # Full batch training
            losses = self._model.train_epochs(X, y, epochs=epochs, verbose=verbose > 0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with AMRC."""
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)
    
    def reset_memory(self):
        """Reset AMRC memory state."""
        self._model.reset_memory()
    
    def get_memory_state(self) -> np.ndarray:
        """Get AMRC memory state."""
        return self._model.get_memory_state()
    
    def set_memory_state(self, memory: np.ndarray):
        """Set AMRC memory state."""
        memory = np.asarray(memory, dtype=np.float32)
        self._model.set_memory_state(memory)
    
    def freeze_hidden_layer(self):
        """Freeze hidden layer weights during training."""
        self._model.freeze_hidden()
    
    def unfreeze_hidden_layer(self):
        """Unfreeze hidden layer weights."""
        self._model.unfreeze_hidden()
    
    def freeze_output_layer(self):
        """Freeze output layer weights during training."""
        self._model.freeze_output()
    
    def unfreeze_output_layer(self):
        """Unfreeze output layer weights."""
        self._model.unfreeze_output()
    
    def save(self, filepath: str):
        """Save AMRC model to file."""
        self._model.save(filepath)
    
    def load(self, filepath: str):
        """Load AMRC model from file."""
        self._model.load(filepath)
    
    @property
    def beta(self) -> float:
        """Get beta (memory preservation) parameter."""
        return self._model.beta
    
    @beta.setter
    def beta(self, value: float):
        """Set beta parameter."""
        self._model.beta = value
    
    @property
    def alpha(self) -> float:
        """Get alpha (memory update rate) parameter."""
        return self._model.alpha
    
    @alpha.setter
    def alpha(self, value: float):
        """Set alpha parameter."""
        self._model.alpha = value
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self._model.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        """Set learning rate."""
        self._model.learning_rate = value
    
    @property
    def training_steps(self) -> int:
        """Get number of training steps."""
        return self._model.training_steps
    
    @property
    def last_loss(self) -> float:
        """Get last training loss."""
        return self._model.last_loss


class PMRC(MemoryCell):
    """
    Persistent Memory Recurrent Cell (PMRC)
    
    An advanced memory-native neural network with learnable memory dynamics.
    Features all AMRC capabilities plus:
    - Learnable memory gates (network learns what to remember)
    - Output layer memory
    - Advanced partial training options
    - Multiple serialization formats
    
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden neurons with memory
    output_size : int
        Number of output neurons
    beta : float, default=0.3
        Hidden layer memory preservation factor (0.0 to 1.0)
    alpha : float, default=0.1
        Memory update rate (0.0 to 1.0)
    output_beta : float, default=0.2
        Output layer memory preservation factor
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    use_recurrent : bool, default=True
        Enable recurrent connections in hidden layer
    use_learnable_gates : bool, default=False
        Enable learnable memory gate parameters
    use_output_memory : bool, default=False
        Enable memory in output layer
    random_state : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> # Create PMRC with learnable gates
    >>> model = PMRC(
    ...     input_size=10, 
    ...     hidden_size=20, 
    ...     output_size=5,
    ...     use_learnable_gates=True
    ... )
    >>> model.fit(X_train, y_train, epochs=50)
    
    >>> # Advanced partial training
    >>> model.freeze_hidden_percentage(0.5)  # Freeze 50% of hidden neurons
    >>> model.freeze_memory_gates()  # Freeze memory dynamics
    >>> model.fit(X_new, y_new, epochs=20)
    
    >>> # Access learnable gate states
    >>> gates = model.get_gate_state()
    >>> print(f"Average gate value: {model.avg_gate_value}")
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
                 use_learnable_gates: bool = False,
                 use_output_memory: bool = False,
                 random_state: Optional[int] = None):
        
        if not HAS_PMRC:
            raise RuntimeError(
                "PMRC implementation not available. "
                "Please compile memory_net_extended.c first."
            )
        
        super().__init__("PMRC")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create underlying model
        self._model = PMRC_Base(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            beta=beta,
            alpha=alpha,
            output_beta=output_beta,
            learning_rate=learning_rate,
            use_recurrent=use_recurrent,
            use_learnable_gates=use_learnable_gates,
            use_output_memory=use_output_memory,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: Optional[int] = None,
            validation_split: float = 0.0,
            verbose: int = 0,
            reset_memory: bool = False) -> 'PMRC':
        """Train the PMRC model."""
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if reset_memory:
            self._model.reset_memory()
        
        # Use sklearn-compatible API
        self._model.fit(X, y, epochs=epochs, verbose=verbose)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with PMRC."""
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        return self._model.score(X, y)
    
    def reset_memory(self):
        """Reset PMRC memory state."""
        self._model.reset_memory()
    
    def get_memory_state(self) -> np.ndarray:
        """Get PMRC memory state."""
        return self._model.get_memory_state()
    
    def set_memory_state(self, memory: np.ndarray):
        """Set PMRC memory state."""
        memory = np.asarray(memory, dtype=np.float32)
        self._model.set_memory_state(memory)
    
    def get_gate_state(self) -> np.ndarray:
        """Get learnable gate states (only if use_learnable_gates=True)."""
        return self._model.get_gate_state()
    
    # Partial training methods
    
    def freeze_hidden_layer(self):
        """Freeze all hidden layer weights."""
        self._model.freeze_hidden_layer()
    
    def unfreeze_hidden_layer(self):
        """Unfreeze all hidden layer weights."""
        self._model.unfreeze_hidden_layer()
    
    def freeze_output_layer(self):
        """Freeze all output layer weights."""
        self._model.freeze_output_layer()
    
    def unfreeze_output_layer(self):
        """Unfreeze all output layer weights."""
        self._model.unfreeze_output_layer()
    
    def freeze_memory_gates(self):
        """Freeze learnable memory gate parameters."""
        self._model.freeze_memory_gates()
    
    def unfreeze_memory_gates(self):
        """Unfreeze learnable memory gate parameters."""
        self._model.unfreeze_memory_gates()
    
    def freeze_hidden_percentage(self, percentage: float):
        """
        Freeze a percentage of hidden neurons.
        
        Parameters
        ----------
        percentage : float
            Percentage to freeze (0.0 to 1.0)
        """
        self._model.freeze_hidden_percentage(percentage)
    
    def freeze_output_percentage(self, percentage: float):
        """
        Freeze a percentage of output neurons.
        
        Parameters
        ----------
        percentage : float
            Percentage to freeze (0.0 to 1.0)
        """
        self._model.freeze_output_percentage(percentage)
    
    def freeze_by_magnitude(self, threshold: float, freeze_large: bool = True):
        """
        Freeze weights based on magnitude.
        
        Parameters
        ----------
        threshold : float
            Weight magnitude threshold
        freeze_large : bool, default=True
            If True, freeze weights with magnitude > threshold
            If False, freeze weights with magnitude < threshold
        """
        self._model.freeze_by_magnitude(threshold, freeze_large)
    
    # Save/Load with multiple formats
    
    def save(self, filepath: str, method: str = 'custom'):
        """
        Save PMRC model to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        method : str, default='custom'
            Serialization method: 'custom', 'joblib', or 'pickle'
        """
        self._model.save_model(filepath, method=method)
    
    def load(self, filepath: str, method: str = 'custom'):
        """
        Load PMRC model from file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        method : str, default='custom'
            Serialization method: 'custom', 'joblib', or 'pickle'
        """
        loaded = PMRC_Base.load_model(filepath, method=method)
        self._model = loaded
    
    # Properties
    
    @property
    def beta(self) -> float:
        """Hidden layer memory preservation parameter."""
        return self._model.beta
    
    @beta.setter
    def beta(self, value: float):
        self._model.beta = value
    
    @property
    def alpha(self) -> float:
        """Memory update rate parameter."""
        return self._model.alpha
    
    @alpha.setter
    def alpha(self, value: float):
        self._model.alpha = value
    
    @property
    def output_beta(self) -> float:
        """Output layer memory preservation parameter."""
        return self._model.output_beta
    
    @output_beta.setter
    def output_beta(self, value: float):
        self._model.output_beta = value
    
    @property
    def learning_rate(self) -> float:
        """Learning rate."""
        return self._model.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self._model.learning_rate = value
    
    @property
    def training_steps(self) -> int:
        """Number of training steps."""
        return self._model.training_steps
    
    @property
    def last_loss(self) -> float:
        """Last training loss."""
        return self._model.last_loss
    
    @property
    def avg_memory_magnitude(self) -> float:
        """Average magnitude of memory values."""
        return self._model.avg_memory_magnitude
    
    @property
    def avg_gate_value(self) -> float:
        """Average value of learnable gates."""
        return self._model.avg_gate_value
    
    @property
    def use_recurrent(self) -> bool:
        """Whether recurrent connections are enabled."""
        return self._model.use_recurrent
    
    @property
    def use_learnable_gates(self) -> bool:
        """Whether learnable gates are enabled."""
        return self._model.use_learnable_gates
    
    @property
    def use_output_memory(self) -> bool:
        """Whether output memory is enabled."""
        return self._model.use_output_memory


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_model(
    model_type: Literal['amrc', 'pmrc', 'AMRC', 'PMRC'],
    input_size: int,
    hidden_size: int,
    output_size: int,
    **kwargs
) -> MemoryCell:
    """
    Factory function to create memory-native neural network models.
    
    Parameters
    ----------
    model_type : {'amrc', 'pmrc', 'AMRC', 'PMRC'}
        Type of model to create
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden neurons
    output_size : int
        Number of output neurons
    **kwargs
        Additional parameters passed to the model constructor
    
    Returns
    -------
    model : MemoryCell
        An instance of AMRC or PMRC
    
    Examples
    --------
    >>> # Create AMRC model
    >>> model = create_model('amrc', 10, 20, 5, beta=0.4)
    
    >>> # Create PMRC model with learnable gates
    >>> model = create_model('pmrc', 10, 20, 5, use_learnable_gates=True)
    """
    model_type = model_type.upper()
    
    if model_type == 'AMRC':
        return AMRC(input_size, hidden_size, output_size, **kwargs)
    elif model_type == 'PMRC':
        return PMRC(input_size, hidden_size, output_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'amrc' or 'pmrc'.")


# ============================================================================
# MODULE INFO
# ============================================================================

def get_info() -> Dict[str, Any]:
    """
    Get information about available models and their status.
    
    Returns
    -------
    info : dict
        Dictionary containing availability and version information
    """
    return {
        'version': __version__,
        'amrc_available': HAS_AMRC,
        'pmrc_available': HAS_PMRC,
        'models': ['AMRC', 'PMRC'],
        'description': 'Memory-Native Neural Networks API'
    }


if __name__ == "__main__":
    # Print module info when run directly
    info = get_info()
    print("\n" + "="*70)
    print("Memory-Native Neural Networks API")
    print("="*70)
    print(f"Version: {info['version']}")
    print(f"\nAvailable Models:")
    print(f"  AMRC (Adaptive Memory Recurrent Cell): {info['amrc_available']}")
    print(f"  PMRC (Persistent Memory Recurrent Cell): {info['pmrc_available']}")
    print("\nFor usage examples, see sample.py")
    print("For detailed documentation, see README.md")
    print("="*70 + "\n")