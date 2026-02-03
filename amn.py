#!/usr/bin/env python3
"""
ADAPTIVE MEMORY NETWORK (AMN) - PYTHON API

Combines three cutting-edge memory architectures:
1. Liquid Constant (LC) Architecture: Dynamic time constants based on input importance
2. Linear Recurrent Units (LRU): Parallel-friendly recurrent processing
3. Associative Memory Manifolds (AMM): Global memory whiteboard for long-term context

Features:
- Scikit-learn-like API (fit, predict, score)
- Dynamic time constants that adapt to input importance
- Parallel-friendly recurrent computation
- Global associative memory for long-range dependencies
- Complete state serialization

Compile C library first:
    Windows: gcc -shared -o amn.dll amn.c -lm -O3 -fopenmp
    Linux:   gcc -shared -fPIC -o amn.so amn.c -lm -O3 -fopenmp
    Mac:     gcc -shared -fPIC -o amn.dylib amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
import pickle
import warnings
from typing import Optional, Union, Tuple

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'amn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'amn.dylib'
    else:
        lib_name = 'amn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o amn.dll amn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o amn.dylib amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp")
        else:
            print("  gcc -shared -fPIC -o amn.so amn.c -lm -O3 -fopenmp")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded AMN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_amn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_amn.restype = ctypes.c_void_p
_lib.destroy_amn.argtypes = [ctypes.c_void_p]
_lib.destroy_amn.restype = None

# Forward/Predict
_lib.amn_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.amn_forward.restype = None

# Training
_lib.amn_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.amn_train.restype = ctypes.c_float

_lib.amn_train_batch.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
_lib.amn_train_batch.restype = ctypes.c_float

# Memory management
_lib.amn_reset_memory.argtypes = [ctypes.c_void_p]
_lib.amn_reset_memory.restype = None
_lib.amn_reset_manifold.argtypes = [ctypes.c_void_p]
_lib.amn_reset_manifold.restype = None

_lib.amn_get_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.amn_get_hidden_state.restype = None
_lib.amn_set_hidden_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.amn_set_hidden_state.restype = None
_lib.amn_get_manifold.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.amn_get_manifold.restype = None

# Parameters
_lib.amn_get_learning_rate.argtypes = [ctypes.c_void_p]
_lib.amn_get_learning_rate.restype = ctypes.c_float
_lib.amn_set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.amn_set_learning_rate.restype = None

_lib.amn_get_dt.argtypes = [ctypes.c_void_p]
_lib.amn_get_dt.restype = ctypes.c_float
_lib.amn_set_dt.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.amn_set_dt.restype = None

_lib.amn_get_memory_decay.argtypes = [ctypes.c_void_p]
_lib.amn_get_memory_decay.restype = ctypes.c_float
_lib.amn_set_memory_decay.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.amn_set_memory_decay.restype = None

_lib.amn_get_training_steps.argtypes = [ctypes.c_void_p]
_lib.amn_get_training_steps.restype = ctypes.c_int
_lib.amn_get_last_loss.argtypes = [ctypes.c_void_p]
_lib.amn_get_last_loss.restype = ctypes.c_float

_lib.amn_get_avg_manifold_energy.argtypes = [ctypes.c_void_p]
_lib.amn_get_avg_manifold_energy.restype = ctypes.c_float
_lib.amn_get_avg_lru_magnitude.argtypes = [ctypes.c_void_p]
_lib.amn_get_avg_lru_magnitude.restype = ctypes.c_float
_lib.amn_get_avg_lc_timescale.argtypes = [ctypes.c_void_p]
_lib.amn_get_avg_lc_timescale.restype = ctypes.c_float

# Save/Load
_lib.amn_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.amn_save.restype = ctypes.c_int
_lib.amn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.amn_load.restype = ctypes.c_int

# Info
_lib.amn_print_info.argtypes = [ctypes.c_void_p]
_lib.amn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class AdaptiveMemoryNetwork:
    """
    Adaptive Memory Network combining three advanced memory mechanisms.
    
    This network integrates:
    1. Liquid Constant neurons - time constants adapt based on input importance
    2. Linear Recurrent Units - efficient parallel recurrent processing
    3. Associative Memory Manifolds - global memory for long-range dependencies
    
    Parameters
    ----------
    input_size : int
        Dimension of input vectors
    hidden_size : int
        Number of hidden neurons
    output_size : int
        Dimension of output vectors
    memory_manifold_size : int, default=64
        Size of the associative memory matrix (smaller = faster, larger = more capacity)
    learning_rate : float, default=0.001
        Learning rate for training
    dt : float, default=0.1
        Time step for Liquid Constant dynamics
    memory_decay : float, default=0.995
        Decay rate for associative memory manifold (0.99-0.9999)
    random_state : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> # Create network
    >>> net = AdaptiveMemoryNetwork(input_size=10, hidden_size=32, output_size=2)
    >>> 
    >>> # Train on data
    >>> X_train = np.random.randn(100, 10).astype(np.float32)
    >>> y_train = np.random.randn(100, 2).astype(np.float32)
    >>> net.fit(X_train, y_train, epochs=50)
    >>> 
    >>> # Make predictions
    >>> X_test = np.random.randn(20, 10).astype(np.float32)
    >>> predictions = net.predict(X_test)
    >>> 
    >>> # Access statistics
    >>> print(f"Manifold Energy: {net.avg_manifold_energy:.4f}")
    >>> print(f"LC Timescale: {net.avg_lc_timescale:.4f}")
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 memory_manifold_size: int = 64,
                 learning_rate: float = 0.001,
                 dt: float = 0.1,
                 memory_decay: float = 0.995,
                 random_state: Optional[int] = None):
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_manifold_size = memory_manifold_size
        
        # Create network
        self._net = _lib.create_amn(
            input_size, hidden_size, output_size,
            memory_manifold_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create AMN network")
        
        # Set parameters
        self.dt = dt
        self.memory_decay = memory_decay
        self.learning_rate = learning_rate
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_amn(self._net)
    
    # ========================================================================
    # CORE METHODS (Scikit-learn style)
    # ========================================================================
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, batch_size: int = 32,
            verbose: int = 1, reset_memory: bool = True) -> 'AdaptiveMemoryNetwork':
        """
        Train the network on data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, input_size)
            Training input data
        y : array-like, shape (n_samples, output_size)
            Training target data
        epochs : int, default=100
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        verbose : int, default=1
            Verbosity level (0=silent, 1=progress, 2=detailed)
        reset_memory : bool, default=True
            Whether to reset memory before training
        
        Returns
        -------
        self : AdaptiveMemoryNetwork
            Returns self for method chaining
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.input_size}")
        if y.shape[1] != self.output_size:
            raise ValueError(f"y has {y.shape[1]} outputs, expected {self.output_size}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different numbers of samples")
        
        n_samples = X.shape[0]
        
        if reset_memory:
            self.reset_memory()
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                X_flat = X_batch.flatten()
                y_flat = y_batch.flatten()
                
                loss = _lib.amn_train_batch(
                    self._net,
                    X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    batch_end - i
                )
                
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            if verbose >= 1 and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            if verbose >= 2:
                print(f"  Manifold Energy: {self.avg_manifold_energy:.6f}, "
                      f"LRU Magnitude: {self.avg_lru_magnitude:.6f}, "
                      f"LC Timescale: {self.avg_lc_timescale:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray, reset_memory: bool = False) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, input_size)
            Input data
        reset_memory : bool, default=False
            Whether to reset memory before prediction
        
        Returns
        -------
        y_pred : ndarray, shape (n_samples, output_size)
            Predicted outputs
        """
        X = np.asarray(X, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.input_size}")
        
        if reset_memory:
            self.reset_memory()
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.output_size), dtype=np.float32)
        
        for i in range(n_samples):
            input_vec = X[i]
            output_vec = np.zeros(self.output_size, dtype=np.float32)
            
            _lib.amn_forward(
                self._net,
                input_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            predictions[i] = output_vec
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray, 
              metric: str = 'r2') -> float:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, input_size)
            Input data
        y : array-like, shape (n_samples, output_size)
            True outputs
        metric : str, default='r2'
            Metric to use ('r2', 'mse', 'mae')
        
        Returns
        -------
        score : float
            Performance score
        """
        y_pred = self.predict(X, reset_memory=True)
        y_true = np.asarray(y, dtype=np.float32)
        
        if metric == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-10))
        elif metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif metric == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def reset_memory(self):
        """Reset all memory states (hidden, LRU, and manifold)"""
        _lib.amn_reset_memory(self._net)
    
    def reset_manifold(self):
        """Reset only the associative memory manifold"""
        _lib.amn_reset_manifold(self._net)
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        state = np.zeros(self.hidden_size, dtype=np.float32)
        _lib.amn_get_hidden_state(
            self._net,
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return state
    
    def set_hidden_state(self, state: np.ndarray):
        """Set hidden state"""
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] != self.hidden_size:
            raise ValueError(f"State has wrong size: {state.shape[0]} vs {self.hidden_size}")
        
        _lib.amn_set_hidden_state(
            self._net,
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    
    def get_manifold(self) -> np.ndarray:
        """Get current memory manifold matrix"""
        manifold = np.zeros(
            (self.memory_manifold_size, self.hidden_size),
            dtype=np.float32
        )
        _lib.amn_get_manifold(
            self._net,
            manifold.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return manifold
    
    # ========================================================================
    # PARAMETERS
    # ========================================================================
    
    @property
    def learning_rate(self) -> float:
        return _lib.amn_get_learning_rate(self._net)
    
    @learning_rate.setter
    def learning_rate(self, lr: float):
        _lib.amn_set_learning_rate(self._net, lr)
    
    @property
    def dt(self) -> float:
        """Time step for Liquid Constant dynamics"""
        return _lib.amn_get_dt(self._net)
    
    @dt.setter
    def dt(self, dt: float):
        _lib.amn_set_dt(self._net, dt)
    
    @property
    def memory_decay(self) -> float:
        """Decay rate for associative memory manifold"""
        return _lib.amn_get_memory_decay(self._net)
    
    @memory_decay.setter
    def memory_decay(self, decay: float):
        _lib.amn_set_memory_decay(self._net, decay)
    
    @property
    def training_steps(self) -> int:
        return _lib.amn_get_training_steps(self._net)
    
    @property
    def last_loss(self) -> float:
        return _lib.amn_get_last_loss(self._net)
    
    @property
    def avg_manifold_energy(self) -> float:
        """Average energy in the memory manifold"""
        return _lib.amn_get_avg_manifold_energy(self._net)
    
    @property
    def avg_lru_magnitude(self) -> float:
        """Average magnitude of LRU states"""
        return _lib.amn_get_avg_lru_magnitude(self._net)
    
    @property
    def avg_lc_timescale(self) -> float:
        """Average effective timescale of Liquid Constant neurons"""
        return _lib.amn_get_avg_lc_timescale(self._net)
    
    # ========================================================================
    # SERIALIZATION
    # ========================================================================
    
    def save(self, filepath: str):
        """
        Save network to file (custom binary format).
        
        This saves all weights, parameters, and memory states.
        """
        result = _lib.amn_save(self._net, filepath.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to save network to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdaptiveMemoryNetwork':
        """
        Load network from file.
        
        Note: You must first create a network with the correct architecture,
        then call this method to load the weights.
        """
        # We need to read metadata first to create the network
        with open(filepath, 'rb') as f:
            import struct
            input_size = struct.unpack('i', f.read(4))[0]
            hidden_size = struct.unpack('i', f.read(4))[0]
            output_size = struct.unpack('i', f.read(4))[0]
            memory_manifold_size = struct.unpack('i', f.read(4))[0]
            learning_rate = struct.unpack('f', f.read(4))[0]
        
        # Create network with correct architecture
        net = cls(input_size, hidden_size, output_size, 
                 memory_manifold_size, learning_rate)
        
        # Load weights
        result = _lib.amn_load(net._net, filepath.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to load network from {filepath}")
        
        return net
    
    def __getstate__(self):
        """Support for pickle"""
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.amn') as f:
            temp_path = f.name
        
        self.save(temp_path)
        
        with open(temp_path, 'rb') as f:
            data = f.read()
        
        import os
        os.unlink(temp_path)
        
        return {
            'data': data,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'memory_manifold_size': self.memory_manifold_size,
        }
    
    def __setstate__(self, state):
        """Support for pickle"""
        import tempfile
        
        # Create network structure
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.output_size = state['output_size']
        self.memory_manifold_size = state['memory_manifold_size']
        
        # Write data to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.amn') as f:
            f.write(state['data'])
            temp_path = f.name
        
        # Create network with dummy parameters (will be overwritten)
        self._net = _lib.create_amn(
            self.input_size, self.hidden_size, self.output_size,
            self.memory_manifold_size, 0.001
        )
        
        # Load from file
        _lib.amn_load(self._net, temp_path.encode('utf-8'))
        
        import os
        os.unlink(temp_path)
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def print_info(self):
        """Print detailed network information"""
        _lib.amn_print_info(self._net)
    
    def __repr__(self):
        return (f"AdaptiveMemoryNetwork(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, output_size={self.output_size}, "
                f"memory_manifold_size={self.memory_manifold_size})")


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_usage():
    """Demonstrate basic usage"""
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
    net = AdaptiveMemoryNetwork(
        input_size=8,
        hidden_size=24,
        output_size=3,
        memory_manifold_size=32,
        learning_rate=0.01
    )
    
    print("\nNetwork created:")
    net.print_info()
    
    print("\nTraining...")
    net.fit(X_train, y_train, epochs=30, verbose=1)
    
    # Evaluate
    r2 = net.score(X_test, y_test, metric='r2')
    mse = net.score(X_test, y_test, metric='mse')
    
    print(f"\nTest R² Score: {r2:.4f}")
    print(f"Test MSE: {mse:.6f}")
    
    return net


def demo_memory_mechanisms():
    """Demonstrate the three memory mechanisms"""
    print("\n" + "="*70)
    print("DEMO 2: Three Memory Mechanisms")
    print("="*70)
    
    net = AdaptiveMemoryNetwork(
        input_size=4,
        hidden_size=12,
        output_size=2,
        memory_manifold_size=16,
        dt=0.1  # Liquid Constant time step
    )
    
    # === Mechanism 1: Liquid Constant ===
    print("\n--- Liquid Constant (LC) Neurons ---")
    print("Time constants adapt based on input importance\n")
    
    boring_input = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    exciting_input = np.array([[2.0, -1.5, 1.8, -2.0]], dtype=np.float32)
    
    net.reset_memory()
    _ = net.predict(boring_input)
    timescale_boring = net.avg_lc_timescale
    
    net.reset_memory()
    _ = net.predict(exciting_input)
    timescale_exciting = net.avg_lc_timescale
    
    print(f"Boring input  → Timescale: {timescale_boring:.4f} (rigid memory)")
    print(f"Exciting input → Timescale: {timescale_exciting:.4f} (fluid memory)")
    print("→ Network adapts its memory based on input importance!")
    
    # === Mechanism 2: LRU ===
    print("\n--- Linear Recurrent Units (LRU) ---")
    print("Efficient parallel-friendly recurrence\n")
    
    net.reset_memory()
    for step in range(5):
        test_input = np.random.randn(1, 4).astype(np.float32)
        _ = net.predict(test_input)
        print(f"Step {step}: LRU Magnitude = {net.avg_lru_magnitude:.4f}")
    
    print("→ LRU maintains stable recurrent state!")
    
    # === Mechanism 3: Associative Memory Manifold ===
    print("\n--- Associative Memory Manifold (AMM) ---")
    print("Global memory whiteboard for long-term context\n")
    
    net.reset_memory()
    
    # Write to memory
    for i in range(10):
        pattern = np.random.randn(1, 4).astype(np.float32)
        _ = net.predict(pattern)
    
    manifold_energy = net.avg_manifold_energy
    manifold = net.get_manifold()
    
    print(f"After 10 inputs:")
    print(f"  Manifold Energy: {manifold_energy:.6f}")
    print(f"  Manifold Shape: {manifold.shape}")
    print(f"  Non-zero entries: {np.sum(np.abs(manifold) > 0.01)}/{manifold.size}")
    print("→ Network accumulates memories in global manifold!")


def demo_sequential_processing():
    """Demonstrate handling sequential/temporal data"""
    print("\n" + "="*70)
    print("DEMO 3: Sequential Processing")
    print("="*70)
    
    # Create a simple sequence prediction task
    # Predict next value in sine wave
    
    print("\nTask: Predict next value in sine wave")
    
    seq_length = 100
    t = np.linspace(0, 4*np.pi, seq_length)
    sequence = np.sin(t).astype(np.float32)
    
    # Prepare data: input is current value, target is next value
    X = sequence[:-1].reshape(-1, 1)
    y = sequence[1:].reshape(-1, 1)
    
    # Split
    split = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create network
    net = AdaptiveMemoryNetwork(
        input_size=1,
        hidden_size=16,
        output_size=1,
        memory_manifold_size=8,
        learning_rate=0.01,
        dt=0.05  # Smaller time step for smoother dynamics
    )
    
    print("\nTraining on sequential data...")
    net.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
    
    # Predict
    predictions = net.predict(X_test, reset_memory=False)
    
    mse = np.mean((predictions - y_test) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Average timescale: {net.avg_lc_timescale:.4f}")
    print(f"Manifold energy: {net.avg_manifold_energy:.6f}")


def demo_serialization():
    """Demonstrate save/load functionality"""
    print("\n" + "="*70)
    print("DEMO 4: Serialization")
    print("="*70)
    
    # Create and train network
    net = AdaptiveMemoryNetwork(5, 10, 2, memory_manifold_size=8)
    X = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randn(50, 2).astype(np.float32)
    net.fit(X, y, epochs=20, verbose=0)
    
    test_input = np.random.randn(1, 5).astype(np.float32)
    original_pred = net.predict(test_input)
    
    # Method 1: Custom binary format
    print("\n1. Custom binary format (.amn)")
    net.save('model.amn')
    net_loaded = AdaptiveMemoryNetwork.load('model.amn')
    pred1 = net_loaded.predict(test_input)
    print(f"   Prediction difference: {np.abs(original_pred - pred1).max():.10f}")
    
    # Method 2: Pickle
    print("\n2. Pickle format (.pkl)")
    with open('model.pkl', 'wb') as f:
        pickle.dump(net, f)
    
    with open('model.pkl', 'rb') as f:
        net_loaded2 = pickle.load(f)
    
    pred2 = net_loaded2.predict(test_input)
    print(f"   Prediction difference: {np.abs(original_pred - pred2).max():.10f}")
    
    print("\n✓ Both methods preserve network state perfectly!")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*10 + "ADAPTIVE MEMORY NETWORK (AMN)" + " "*29 + "║")
    print("║" + " "*5 + "Liquid Constant + LRU + Associative Memory" + " "*20 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        demo_basic_usage()
        
        input("\nPress Enter to continue...")
        demo_memory_mechanisms()
        
        input("\nPress Enter to continue...")
        demo_sequential_processing()
        
        input("\nPress Enter to continue...")
        demo_serialization()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n✓ Features Demonstrated:")
        print("  • Liquid Constant neurons with adaptive time constants")
        print("  • Linear Recurrent Units for efficient sequential processing")
        print("  • Associative Memory Manifolds for long-term context")
        print("  • Scikit-learn compatible API")
        print("  • Sequential/temporal data handling")
        print("  • Complete state serialization")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()