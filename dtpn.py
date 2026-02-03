#!/usr/bin/env python3
"""
DUAL-TRACK PERSISTENCE NETWORK (DTPN) - PYTHON API

Combines three memory persistence mechanisms:

CONCEPT 1: Memory-Preserving Activation
  y(t) = activation(W·x(t) + U·h(t-1)) + β·y(t-1)
  Creates a mathematical "echo" of past computations

CONCEPT 2: Stateful Neurons
  memory(t) = (1-α)·memory(t-1) + α·new_info
  Persistent internal state with exponential decay

CONCEPT 3: Global Memory Matrix
  M_t = M_{t-1} + σ(x_t·K^T)V
  External storage for long-term context

Usage:
  1. Compile C library:
     Windows: gcc -shared -o dtpn.dll dtpn.c -lm -O3
     Linux:   gcc -shared -fPIC -o dtpn.so dtpn.c -lm -O3
     Mac:     gcc -shared -fPIC -o dtpn.dylib dtpn.c -lm -O3
  
  2. Import and use:
     from dtpn import DualTrackPersistenceNetwork
     net = DualTrackPersistenceNetwork(input_size=10, hidden_size=32, output_size=2)
     net.fit(X_train, y_train, epochs=50)
     predictions = net.predict(X_test)

License: GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
import pickle
from typing import Optional, Union, Tuple

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'dtpn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'dtpn.dylib'
    else:
        lib_name = 'dtpn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o dtpn.dll dtpn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o dtpn.dylib dtpn.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o dtpn.so dtpn.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded DTPN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_dtpn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_dtpn.restype = ctypes.c_void_p
_lib.destroy_dtpn.argtypes = [ctypes.c_void_p]
_lib.destroy_dtpn.restype = None

# Forward/Predict
_lib.dtpn_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.dtpn_forward.restype = None

# Training
_lib.dtpn_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.dtpn_train.restype = ctypes.c_float

_lib.dtpn_train_batch.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
_lib.dtpn_train_batch.restype = ctypes.c_float

# Memory management
_lib.dtpn_reset_memory.argtypes = [ctypes.c_void_p]
_lib.dtpn_reset_memory.restype = None
_lib.dtpn_reset_global_memory.argtypes = [ctypes.c_void_p]
_lib.dtpn_reset_global_memory.restype = None

_lib.dtpn_get_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.dtpn_get_hidden_state.restype = None
_lib.dtpn_set_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.dtpn_set_hidden_state.restype = None
_lib.dtpn_get_memory_matrix.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.dtpn_get_memory_matrix.restype = None

# Parameters
_lib.dtpn_get_beta.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_beta.restype = ctypes.c_float
_lib.dtpn_set_beta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.dtpn_set_beta.restype = None

_lib.dtpn_get_alpha.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_alpha.restype = ctypes.c_float
_lib.dtpn_set_alpha.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.dtpn_set_alpha.restype = None

_lib.dtpn_get_memory_decay.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_memory_decay.restype = ctypes.c_float
_lib.dtpn_set_memory_decay.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.dtpn_set_memory_decay.restype = None

_lib.dtpn_get_learning_rate.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_learning_rate.restype = ctypes.c_float
_lib.dtpn_set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.dtpn_set_learning_rate.restype = None

_lib.dtpn_get_training_steps.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_training_steps.restype = ctypes.c_int
_lib.dtpn_get_last_loss.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_last_loss.restype = ctypes.c_float

_lib.dtpn_get_avg_echo_magnitude.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_avg_echo_magnitude.restype = ctypes.c_float
_lib.dtpn_get_avg_memory_magnitude.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_avg_memory_magnitude.restype = ctypes.c_float
_lib.dtpn_get_avg_matrix_energy.argtypes = [ctypes.c_void_p]
_lib.dtpn_get_avg_matrix_energy.restype = ctypes.c_float

# Save/Load
_lib.dtpn_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.dtpn_save.restype = ctypes.c_int
_lib.dtpn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.dtpn_load.restype = ctypes.c_int

# Info
_lib.dtpn_print_info.argtypes = [ctypes.c_void_p]
_lib.dtpn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class DualTrackPersistenceNetwork:
    """
    Dual-Track Persistence Network with three memory mechanisms.
    
    This network integrates:
    1. Memory-Preserving Activation - output echo from previous timestep
    2. Stateful Neurons - persistent internal memory per neuron
    3. Global Memory Matrix - external long-term storage
    
    Parameters
    ----------
    input_size : int
        Dimension of input vectors
    hidden_size : int
        Number of hidden neurons
    output_size : int
        Dimension of output vectors
    memory_matrix_size : int, default=32
        Size of global memory matrix (rows)
    learning_rate : float, default=0.001
        Learning rate for training
    beta : float, default=0.3
        Memory preservation factor (Concept 1: echo strength)
    alpha : float, default=0.1
        Memory update rate (Concept 2: how fast neuron memory updates)
    memory_decay : float, default=0.99
        Decay rate for global memory matrix (Concept 3)
    random_state : int, optional
        Random seed for reproducibility
    
    Attributes
    ----------
    avg_echo_magnitude : float
        Average magnitude of output echo (Concept 1)
    avg_memory_magnitude : float
        Average magnitude of neuron internal memory (Concept 2)
    avg_matrix_energy : float
        Average energy in global memory matrix (Concept 3)
    
    Examples
    --------
    >>> # Create network
    >>> net = DualTrackPersistenceNetwork(input_size=10, hidden_size=32, output_size=2)
    >>> 
    >>> # Train on sequential data
    >>> X_train = np.random.randn(100, 10).astype(np.float32)
    >>> y_train = np.random.randn(100, 2).astype(np.float32)
    >>> net.fit(X_train, y_train, epochs=50)
    >>> 
    >>> # Make predictions
    >>> predictions = net.predict(X_test)
    >>> 
    >>> # View memory statistics
    >>> print(f"Echo Magnitude: {net.avg_echo_magnitude:.4f}")
    >>> print(f"Neuron Memory: {net.avg_memory_magnitude:.4f}")
    >>> print(f"Matrix Energy: {net.avg_matrix_energy:.4f}")
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 memory_matrix_size: int = 32,
                 learning_rate: float = 0.001,
                 beta: float = 0.3,
                 alpha: float = 0.1,
                 memory_decay: float = 0.99,
                 random_state: Optional[int] = None):
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_matrix_size = memory_matrix_size
        
        # Create network
        self._net = _lib.create_dtpn(
            input_size, hidden_size, output_size,
            memory_matrix_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create DTPN network")
        
        # Set parameters
        self.beta = beta
        self.alpha = alpha
        self.memory_decay = memory_decay
        self.learning_rate = learning_rate
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_dtpn(self._net)
    
    # === PROPERTIES ===
    
    @property
    def beta(self) -> float:
        """Memory preservation factor (Concept 1)"""
        return _lib.dtpn_get_beta(self._net)
    
    @beta.setter
    def beta(self, value: float):
        _lib.dtpn_set_beta(self._net, value)
    
    @property
    def alpha(self) -> float:
        """Memory update rate (Concept 2)"""
        return _lib.dtpn_get_alpha(self._net)
    
    @alpha.setter
    def alpha(self, value: float):
        _lib.dtpn_set_alpha(self._net, value)
    
    @property
    def memory_decay(self) -> float:
        """Global memory matrix decay (Concept 3)"""
        return _lib.dtpn_get_memory_decay(self._net)
    
    @memory_decay.setter
    def memory_decay(self, value: float):
        _lib.dtpn_set_memory_decay(self._net, value)
    
    @property
    def learning_rate(self) -> float:
        """Learning rate"""
        return _lib.dtpn_get_learning_rate(self._net)
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        _lib.dtpn_set_learning_rate(self._net, value)
    
    @property
    def training_steps(self) -> int:
        """Number of training steps performed"""
        return _lib.dtpn_get_training_steps(self._net)
    
    @property
    def last_loss(self) -> float:
        """Most recent training loss"""
        return _lib.dtpn_get_last_loss(self._net)
    
    @property
    def avg_echo_magnitude(self) -> float:
        """Average magnitude of output echo (Concept 1)"""
        return _lib.dtpn_get_avg_echo_magnitude(self._net)
    
    @property
    def avg_memory_magnitude(self) -> float:
        """Average magnitude of neuron internal memory (Concept 2)"""
        return _lib.dtpn_get_avg_memory_magnitude(self._net)
    
    @property
    def avg_matrix_energy(self) -> float:
        """Average energy in global memory matrix (Concept 3)"""
        return _lib.dtpn_get_avg_matrix_energy(self._net)
    
    # === CORE METHODS ===
    
    def predict(self, X: np.ndarray, reset_memory: bool = True) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)
            Input data
        reset_memory : bool, default=True
            Whether to reset all memory states before prediction
        
        Returns
        -------
        predictions : np.ndarray, shape (n_samples, output_size)
            Predicted outputs
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {X.shape[1]}")
        
        if reset_memory:
            self.reset_memory()
        
        predictions = np.zeros((X.shape[0], self.output_size), dtype=np.float32)
        
        for i, x in enumerate(X):
            x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            pred_ptr = predictions[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _lib.dtpn_forward(self._net, x_ptr, pred_ptr)
        
        return predictions
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100,
            batch_size: int = 32,
            reset_memory_each_epoch: bool = True,
            verbose: int = 0) -> 'DualTrackPersistenceNetwork':
        """
        Train the network on data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)
            Training input data
        y : np.ndarray, shape (n_samples, output_size)
            Training target data
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        reset_memory_each_epoch : bool, default=True
            Whether to reset memory at the start of each epoch
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Returns
        -------
        self : DualTrackPersistenceNetwork
            Returns self for method chaining
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            if reset_memory_each_epoch:
                self.reset_memory()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                X_ptr = X_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                y_ptr = y_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                loss = _lib.dtpn_train_batch(self._net, X_ptr, y_ptr, len(X_batch))
                epoch_loss += loss
            
            epoch_loss /= n_batches
            
            if verbose >= 1 and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f} "
                      f"- Echo: {self.avg_echo_magnitude:.4f} "
                      f"- Memory: {self.avg_memory_magnitude:.4f} "
                      f"- Matrix: {self.avg_matrix_energy:.4f}")
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: str = 'mse') -> float:
        """
        Calculate score on test data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_size)
            Test input data
        y : np.ndarray, shape (n_samples, output_size)
            Test target data
        metric : str, default='mse'
            Metric to use ('mse', 'mae', 'r2')
        
        Returns
        -------
        score : float
            Calculated score
        """
        predictions = self.predict(X)
        y = np.asarray(y, dtype=np.float32)
        
        if metric == 'mse':
            return np.mean((predictions - y) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(predictions - y))
        elif metric == 'r2':
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # === MEMORY MANAGEMENT ===
    
    def reset_memory(self):
        """Reset all memory states (echo, neuron memory, global matrix)"""
        _lib.dtpn_reset_memory(self._net)
    
    def reset_global_memory(self):
        """Reset only the global memory matrix (Concept 3)"""
        _lib.dtpn_reset_global_memory(self._net)
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        state = np.zeros(self.hidden_size, dtype=np.float32)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.dtpn_get_hidden_state(self._net, state_ptr)
        return state
    
    def set_hidden_state(self, state: np.ndarray):
        """Set hidden state"""
        state = np.asarray(state, dtype=np.float32)
        if len(state) != self.hidden_size:
            raise ValueError(f"Expected state of size {self.hidden_size}")
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.dtpn_set_hidden_state(self._net, state_ptr)
    
    def get_memory_matrix(self) -> np.ndarray:
        """Get current global memory matrix (Concept 3)"""
        matrix = np.zeros((self.memory_matrix_size, self.hidden_size), dtype=np.float32)
        matrix_ptr = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.dtpn_get_memory_matrix(self._net, matrix_ptr)
        return matrix
    
    # === SERIALIZATION ===
    
    def save(self, filename: str) -> int:
        """
        Save network to binary file.
        
        Parameters
        ----------
        filename : str
            Path to save file
        
        Returns
        -------
        status : int
            0 on success, -1 on failure
        """
        return _lib.dtpn_save(self._net, filename.encode('utf-8'))
    
    @classmethod
    def load(cls, filename: str) -> 'DualTrackPersistenceNetwork':
        """
        Load network from binary file.
        
        Parameters
        ----------
        filename : str
            Path to saved file
        
        Returns
        -------
        network : DualTrackPersistenceNetwork
            Loaded network
        """
        # Read metadata first to get dimensions
        with open(filename, 'rb') as f:
            import struct
            input_size = struct.unpack('i', f.read(4))[0]
            hidden_size = struct.unpack('i', f.read(4))[0]
            output_size = struct.unpack('i', f.read(4))[0]
            memory_matrix_size = struct.unpack('i', f.read(4))[0]
        
        # Create network with same dimensions
        net = cls(input_size, hidden_size, output_size, memory_matrix_size)
        
        # Load weights and states
        result = _lib.dtpn_load(net._net, filename.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to load network (error code {result})")
        
        return net
    
    def __getstate__(self):
        """Pickle support"""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dtpn') as f:
            temp_path = f.name
        
        self.save(temp_path)
        with open(temp_path, 'rb') as f:
            data = f.read()
        
        import os
        os.unlink(temp_path)
        
        return {
            'binary_data': data,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'memory_matrix_size': self.memory_matrix_size
        }
    
    def __setstate__(self, state):
        """Unpickle support"""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dtpn') as f:
            f.write(state['binary_data'])
            temp_path = f.name
        
        # Create network
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.output_size = state['output_size']
        self.memory_matrix_size = state['memory_matrix_size']
        
        self._net = _lib.create_dtpn(
            self.input_size, self.hidden_size, self.output_size,
            self.memory_matrix_size, 0.001
        )
        
        # Load from file
        _lib.dtpn_load(self._net, temp_path.encode('utf-8'))
        
        import os
        os.unlink(temp_path)
    
    # === INFO ===
    
    def print_info(self):
        """Print network information"""
        _lib.dtpn_print_info(self._net)
    
    def __repr__(self):
        return (f"DualTrackPersistenceNetwork(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, output_size={self.output_size}, "
                f"memory_matrix_size={self.memory_matrix_size})")

# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_usage():
    """Demonstrate basic DTPN usage"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(200, 8).astype(np.float32)
    y = np.random.randn(200, 3).astype(np.float32)
    
    # Split data
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create network
    net = DualTrackPersistenceNetwork(
        input_size=8,
        hidden_size=24,
        output_size=3,
        memory_matrix_size=16,
        learning_rate=0.01,
        beta=0.3,   # Echo preservation
        alpha=0.1,  # Memory update rate
        memory_decay=0.99
    )
    
    print("\nNetwork created:")
    net.print_info()
    
    print("\nTraining...")
    net.fit(X_train, y_train, epochs=30, verbose=1)
    
    # Evaluate
    mse = net.score(X_test, y_test, metric='mse')
    r2 = net.score(X_test, y_test, metric='r2')
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R²: {r2:.4f}")
    
    return net


def demo_three_concepts():
    """Demonstrate the three memory concepts"""
    print("\n" + "="*70)
    print("DEMO 2: Three Memory Concepts")
    print("="*70)
    
    net = DualTrackPersistenceNetwork(
        input_size=4,
        hidden_size=12,
        output_size=2,
        memory_matrix_size=8,
        beta=0.3,
        alpha=0.1,
        memory_decay=0.99
    )
    
    # === Concept 1: Memory-Preserving Activation ===
    print("\n--- CONCEPT 1: Memory-Preserving Activation ---")
    print("y(t) = activation(W·x(t) + U·h(t-1)) + β·y(t-1)")
    print(f"Current β (echo strength): {net.beta}\n")
    
    net.reset_memory()
    test_input = np.array([[1.0, 0.5, -0.3, 0.8]], dtype=np.float32)
    
    print("Processing same input 3 times:")
    for i in range(3):
        _ = net.predict(test_input, reset_memory=False)
        print(f"  Step {i+1}: Echo Magnitude = {net.avg_echo_magnitude:.6f}")
    
    print("\n→ Output echo preserves information from previous timesteps!")
    
    # === Concept 2: Stateful Neurons ===
    print("\n--- CONCEPT 2: Stateful Neurons ---")
    print("memory(t) = (1-α)·memory(t-1) + α·new_info")
    print(f"Current α (update rate): {net.alpha}\n")
    
    net.reset_memory()
    
    print("Processing sequence of inputs:")
    for i in range(5):
        random_input = np.random.randn(1, 4).astype(np.float32)
        _ = net.predict(random_input, reset_memory=False)
        print(f"  Step {i+1}: Neuron Memory = {net.avg_memory_magnitude:.6f}")
    
    print("\n→ Each neuron maintains persistent internal state!")
    
    # === Concept 3: Global Memory Matrix ===
    print("\n--- CONCEPT 3: Global Memory Matrix ---")
    print("M_t = M_{t-1} + σ(x_t·K^T)V")
    print(f"Current decay: {net.memory_decay}\n")
    
    net.reset_memory()
    
    print("Accumulating patterns in global memory:")
    for i in range(10):
        pattern = np.random.randn(1, 4).astype(np.float32)
        _ = net.predict(pattern, reset_memory=False)
        if i % 2 == 0:
            matrix = net.get_memory_matrix()
            print(f"  After {i+1} inputs: Matrix Energy = {net.avg_matrix_energy:.6f}, "
                  f"Active entries = {np.sum(np.abs(matrix) > 0.01)}/{matrix.size}")
    
    print("\n→ Global memory accumulates long-term context!")


def demo_sequential_task():
    """Demonstrate on a sequential prediction task"""
    print("\n" + "="*70)
    print("DEMO 3: Sequential Sine Wave Prediction")
    print("="*70)
    
    # Generate sine wave
    t = np.linspace(0, 4*np.pi, 100)
    sequence = np.sin(t).astype(np.float32)
    
    # Create input-target pairs
    X = sequence[:-1].reshape(-1, 1)
    y = sequence[1:].reshape(-1, 1)
    
    # Split
    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create network
    net = DualTrackPersistenceNetwork(
        input_size=1,
        hidden_size=16,
        output_size=1,
        memory_matrix_size=8,
        learning_rate=0.01,
        beta=0.4,    # Stronger echo for temporal patterns
        alpha=0.15,  # Moderate memory update
        memory_decay=0.95
    )
    
    print("\nTraining on sine wave sequence...")
    net.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
    
    # Predict
    predictions = net.predict(X_test, reset_memory=False)
    
    mse = np.mean((predictions - y_test) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    print(f"\nMemory Statistics:")
    print(f"  Echo Magnitude: {net.avg_echo_magnitude:.6f}")
    print(f"  Neuron Memory:  {net.avg_memory_magnitude:.6f}")
    print(f"  Matrix Energy:  {net.avg_matrix_energy:.6f}")


def demo_hyperparameter_effects():
    """Demonstrate the effect of hyperparameters"""
    print("\n" + "="*70)
    print("DEMO 4: Hyperparameter Effects")
    print("="*70)
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randn(50, 2).astype(np.float32)
    
    configs = [
        ("Low Echo (β=0.1)", {'beta': 0.1, 'alpha': 0.1, 'memory_decay': 0.99}),
        ("High Echo (β=0.5)", {'beta': 0.5, 'alpha': 0.1, 'memory_decay': 0.99}),
        ("Fast Memory (α=0.5)", {'beta': 0.3, 'alpha': 0.5, 'memory_decay': 0.99}),
        ("Slow Decay (0.999)", {'beta': 0.3, 'alpha': 0.1, 'memory_decay': 0.999}),
    ]
    
    print("\nComparing different hyperparameter settings:\n")
    
    for name, params in configs:
        net = DualTrackPersistenceNetwork(
            input_size=5, hidden_size=16, output_size=2,
            memory_matrix_size=8, learning_rate=0.01,
            **params
        )
        
        net.fit(X, y, epochs=20, verbose=0)
        
        # Process test input
        test_input = np.random.randn(1, 5).astype(np.float32)
        _ = net.predict(test_input)
        
        print(f"{name:20s} → Echo: {net.avg_echo_magnitude:.4f}, "
              f"Memory: {net.avg_memory_magnitude:.4f}, "
              f"Matrix: {net.avg_matrix_energy:.4f}")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "DUAL-TRACK PERSISTENCE NETWORK (DTPN)" + " "*21 + "║")
    print("║" + " "*6 + "Memory Echo + Stateful Neurons + Global Matrix" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        demo_basic_usage()
        
        input("\nPress Enter to continue...")
        demo_three_concepts()
        
        input("\nPress Enter to continue...")
        demo_sequential_task()
        
        input("\nPress Enter to continue...")
        demo_hyperparameter_effects()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Three Memory Concepts:")
        print("  [1] Memory-Preserving Activation: y(t) = activation(...) + β·y(t-1)")
        print("  [2] Stateful Neurons: memory(t) = (1-α)·memory(t-1) + α·new_info")
        print("  [3] Global Memory Matrix: M_t = M_{t-1} + σ(x_t·K^T)V")
        
        print("\n✓ Features:")
        print("  • Scikit-learn compatible API (fit, predict, score)")
        print("  • Configurable memory persistence (β, α, decay)")
        print("  • Sequential/temporal data processing")
        print("  • Complete state serialization")
        print("  • Memory statistics and monitoring")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()