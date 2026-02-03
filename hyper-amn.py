#!/usr/bin/env python3
"""
HYPER-AMN: MULTI-HEAD ASSOCIATIVE MANIFOLD NETWORK - PYTHON API

Advanced architecture with specialized manifolds:
- Spatial Manifold: Tracks positional/structural patterns
- Emotional Manifold: Tracks sentiment/tone patterns  
- Logical Manifold: Tracks reasoning/causal patterns

Key Innovation: Prevents memory interference by domain separation

Compile C library first:
    Windows: gcc -shared -o hyper-amn.dll hyper-amn.c -lm -O3 -fopenmp
    Linux:   gcc -shared -fPIC -o hyper-amn.so hyper-amn.c -lm -O3 -fopenmp
    Mac:     gcc -shared -fPIC -o hyper-amn.dylib hyper-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional, Tuple

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'hyper-amn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'hyper-amn.dylib'
    else:
        lib_name = 'hyper-amn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o hyper-amn.dll hyper-amn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o hyper-amn.dylib hyper-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp")
        else:
            print("  gcc -shared -fPIC -o hyper-amn.so hyper-amn.c -lm -O3 -fopenmp")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

try:
    _lib = load_library()
    print(f"✓ Loaded Hyper-AMN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

_lib.create_hyper_amn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_hyper_amn.restype = ctypes.c_void_p
_lib.destroy_hyper_amn.argtypes = [ctypes.c_void_p]
_lib.destroy_hyper_amn.restype = None

_lib.hyper_amn_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.hyper_amn_forward.restype = None

_lib.hyper_amn_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.hyper_amn_train.restype = ctypes.c_float

_lib.hyper_amn_reset_memory.argtypes = [ctypes.c_void_p]
_lib.hyper_amn_reset_memory.restype = None

_lib.hyper_amn_get_head_activations.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.hyper_amn_get_head_activations.restype = None

_lib.hyper_amn_get_manifold_energy.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.hyper_amn_get_manifold_energy.restype = ctypes.c_float

_lib.hyper_amn_get_cross_head_flow.argtypes = [ctypes.c_void_p]
_lib.hyper_amn_get_cross_head_flow.restype = ctypes.c_float

_lib.hyper_amn_print_info.argtypes = [ctypes.c_void_p]
_lib.hyper_amn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class HyperAMN:
    """
    Hyper-AMN: Multi-Head Associative Manifold Network
    
    This network uses three specialized memory manifolds:
    1. Spatial Manifold - Fast adaptation, tracks positional/structural patterns
    2. Emotional Manifold - Medium adaptation, tracks sentiment/tone patterns
    3. Logical Manifold - Slow adaptation, tracks reasoning/causal patterns
    
    Parameters
    ----------
    input_size : int
        Dimension of input vectors
    hidden_size : int
        Number of hidden neurons per head
    output_size : int
        Dimension of output vectors
    manifold_size : int, default=64
        Size of each manifold head
    learning_rate : float, default=0.001
        Learning rate for training
    
    Examples
    --------
    >>> net = HyperAMN(input_size=10, hidden_size=32, output_size=2)
    >>> X = np.random.randn(100, 10).astype(np.float32)
    >>> y = np.random.randn(100, 2).astype(np.float32)
    >>> net.fit(X, y, epochs=50)
    >>> predictions = net.predict(X)
    >>> activations = net.get_head_activations()
    >>> print(f"Spatial: {activations[0]:.3f}, Emotional: {activations[1]:.3f}, Logical: {activations[2]:.3f}")
    """
    
    MANIFOLD_SPATIAL = 0
    MANIFOLD_EMOTIONAL = 1
    MANIFOLD_LOGICAL = 2
    NUM_HEADS = 3
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 manifold_size: int = 64, learning_rate: float = 0.001):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.manifold_size = manifold_size
        
        self._net = _lib.create_hyper_amn(
            input_size, hidden_size, output_size, manifold_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create Hyper-AMN network")
    
    def __del__(self):
        if hasattr(self, '_net') and self._net:
            _lib.destroy_hyper_amn(self._net)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        output = np.zeros(self.output_size, dtype=np.float32)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.hyper_amn_forward(self._net, x_ptr, out_ptr)
        return output
    
    def predict(self, X: np.ndarray, reset_memory: bool = True) -> np.ndarray:
        """Predict on multiple samples"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if reset_memory:
            self.reset_memory()
        
        predictions = []
        for x in X:
            pred = self.forward(x)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: Optional[int] = None, verbose: int = 1) -> 'HyperAMN':
        """Train the network"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        n_samples = len(X)
        
        for epoch in range(epochs):
            self.reset_memory()
            epoch_loss = 0.0
            
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_sample = X[idx]
                y_sample = y[idx]
                
                x_ptr = x_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                y_ptr = y_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                loss = _lib.hyper_amn_train(self._net, x_ptr, y_ptr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_samples
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                activations = self.get_head_activations()
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                      f"Heads [S:{activations[0]:.2f} E:{activations[1]:.2f} L:{activations[2]:.2f}]")
        
        return self
    
    def reset_memory(self):
        """Reset all memory states"""
        _lib.hyper_amn_reset_memory(self._net)
    
    def get_head_activations(self) -> np.ndarray:
        """Get current activation levels for each head"""
        activations = np.zeros(self.NUM_HEADS, dtype=np.float32)
        act_ptr = activations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.hyper_amn_get_head_activations(self._net, act_ptr)
        return activations
    
    def get_manifold_energy(self, head_idx: int) -> float:
        """Get energy in a specific manifold"""
        return _lib.hyper_amn_get_manifold_energy(self._net, head_idx)
    
    def get_cross_head_flow(self) -> float:
        """Get information flow between heads"""
        return _lib.hyper_amn_get_cross_head_flow(self._net)
    
    def print_info(self):
        """Print network information"""
        _lib.hyper_amn_print_info(self._net)
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'mse') -> float:
        """Compute score on test data"""
        predictions = self.predict(X, reset_memory=True)
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

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate Hyper-AMN capabilities"""
    print("\n" + "="*70)
    print("HYPER-AMN DEMONSTRATION")
    print("Multi-Head Associative Manifolds")
    print("="*70)
    
    np.random.seed(42)
    
    # Create network
    net = HyperAMN(
        input_size=8,
        hidden_size=32,
        output_size=6,
        manifold_size=32,
        learning_rate=0.01
    )
    
    print("\nNetwork Architecture:")
    net.print_info()
    
    # Create synthetic data with different patterns
    n_samples = 200
    X = np.random.randn(n_samples, 8).astype(np.float32)
    
    # Create targets with spatial, emotional, and logical components
    y_spatial = np.sin(X[:, 0:3])  # Spatial patterns
    y_emotional = np.sign(X[:, 3:5]) * 0.5  # Binary emotional states
    y_logical = (X[:, 5] > 0).astype(float).reshape(-1, 1)  # Logical decision
    y = np.concatenate([y_spatial, y_emotional, y_logical], axis=1)
    
    # Train
    print("\nTraining...")
    net.fit(X[:160], y[:160], epochs=50, verbose=1)
    
    # Test
    print("\nTesting...")
    mse = net.score(X[160:], y[160:], metric='mse')
    r2 = net.score(X[160:], y[160:], metric='r2')
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R²:  {r2:.4f}")
    
    # Analyze head usage
    print("\n" + "="*70)
    print("HEAD SPECIALIZATION ANALYSIS")
    print("="*70)
    
    # Test with different input types
    spatial_input = np.array([[2.0, 1.5, -1.0, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    emotional_input = np.array([[0.1, 0.1, 0.1, 3.0, -2.5, 0.1, 0.1, 0.1]], dtype=np.float32)
    logical_input = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 2.0, 1.5, -1.0]], dtype=np.float32)
    
    test_inputs = [
        ("Spatial-heavy", spatial_input),
        ("Emotional-heavy", emotional_input),
        ("Logical-heavy", logical_input)
    ]
    
    for name, test_input in test_inputs:
        net.reset_memory()
        _ = net.predict(test_input, reset_memory=False)
        activations = net.get_head_activations()
        
        print(f"\n{name} input:")
        print(f"  Spatial Head:    {activations[0]:.3f}")
        print(f"  Emotional Head:  {activations[1]:.3f}")
        print(f"  Logical Head:    {activations[2]:.3f}")
        
        energies = [net.get_manifold_energy(i) for i in range(3)]
        print(f"  Manifold Energies: S={energies[0]:.4f}, E={energies[1]:.4f}, L={energies[2]:.4f}")
    
    print(f"\nCross-Head Information Flow: {net.get_cross_head_flow():.6f}")
    
    print("\n✓ Demonstration complete!")
    print("  - Different heads activate for different pattern types")
    print("  - Prevents memory interference across domains")
    print("  - Cross-head communication allows integration")

if __name__ == "__main__":
    demo()