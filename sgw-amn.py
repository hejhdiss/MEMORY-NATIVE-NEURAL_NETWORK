#!/usr/bin/env python3
"""
SPARSE GLOBAL WORKSPACE - ADAPTIVE MEMORY NETWORK (SGW-AMN) - PYTHON API

Inspired by Global Workspace Theory of consciousness:
- Thousands of neurons compete to access a tiny "conscious" bottleneck
- Only the most vital information survives compression
- Creates high-level "conscious" summary of data

Key Innovation: Attention by Compression - forces the model to extract
only the most essential features through competitive routing

Compile C library first:
    Windows: gcc -shared -o sgw-amn.dll sgw-amn.c -lm -O3 -fopenmp
    Linux:   gcc -shared -fPIC -o sgw-amn.so sgw-amn.c -lm -O3 -fopenmp
    Mac:     gcc -shared -fPIC -o sgw-amn.dylib sgw-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'sgw-amn.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'sgw-amn.dylib'
    else:
        lib_name = 'sgw-amn.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o sgw-amn.dll sgw-amn.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o sgw-amn.dylib sgw-amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp")
        else:
            print("  gcc -shared -fPIC -o sgw-amn.so sgw-amn.c -lm -O3 -fopenmp")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

try:
    _lib = load_library()
    print(f"✓ Loaded SGW-AMN C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

_lib.create_sgw_amn.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_sgw_amn.restype = ctypes.c_void_p
_lib.destroy_sgw_amn.argtypes = [ctypes.c_void_p]
_lib.destroy_sgw_amn.restype = None

_lib.sgw_amn_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.sgw_amn_forward.restype = None

_lib.sgw_amn_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.sgw_amn_train.restype = ctypes.c_float

_lib.sgw_amn_reset_memory.argtypes = [ctypes.c_void_p]
_lib.sgw_amn_reset_memory.restype = None

_lib.sgw_amn_get_workspace_sparsity.argtypes = [ctypes.c_void_p]
_lib.sgw_amn_get_workspace_sparsity.restype = ctypes.c_float

_lib.sgw_amn_get_competition_entropy.argtypes = [ctypes.c_void_p]
_lib.sgw_amn_get_competition_entropy.restype = ctypes.c_float

_lib.sgw_amn_get_bottleneck_rate.argtypes = [ctypes.c_void_p]
_lib.sgw_amn_get_bottleneck_rate.restype = ctypes.c_float

_lib.sgw_amn_print_info.argtypes = [ctypes.c_void_p]
_lib.sgw_amn_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class SparseGlobalWorkspace:
    """
    Sparse Global Workspace - Adaptive Memory Network (SGW-AMN)
    
    Architecture inspired by Global Workspace Theory of consciousness:
    
    Input → [Large Pre-Bottleneck] → [Tiny Workspace] → [Large Post-Bottleneck] → Output
              (256 neurons)          (16 neurons)         (256 neurons)
    
    Neurons compete to access the workspace bottleneck. Only the most
    important information passes through, forcing the network to extract
    the essence of the data.
    
    Features:
    - Competitive routing to workspace via softmax competition
    - Sparse bottleneck forces attention-by-compression
    - Liquid constant dynamics in the workspace
    - Associative memory manifold for long-term patterns
    
    Parameters
    ----------
    input_size : int
        Dimension of input vectors
    pre_bottleneck_size : int, default=256
        Number of neurons before bottleneck (larger = richer representation)
    workspace_size : int, default=16
        Size of the "conscious" bottleneck (smaller = more selective)
    post_bottleneck_size : int, default=256
        Number of neurons after bottleneck
    output_size : int
        Dimension of output vectors
    manifold_size : int, default=64
        Size of associative memory
    learning_rate : float, default=0.001
        Learning rate for training
    
    Examples
    --------
    >>> net = SparseGlobalWorkspace(
    ...     input_size=10,
    ...     pre_bottleneck_size=128,
    ...     workspace_size=8,
    ...     post_bottleneck_size=128,
    ...     output_size=2
    ... )
    >>> X = np.random.randn(100, 10).astype(np.float32)
    >>> y = np.random.randn(100, 2).astype(np.float32)
    >>> net.fit(X, y, epochs=50)
    >>> print(f"Workspace sparsity: {net.workspace_sparsity:.2%}")
    >>> print(f"Competition entropy: {net.competition_entropy:.4f}")
    """
    
    def __init__(self, input_size: int,
                 pre_bottleneck_size: int = 256,
                 workspace_size: int = 16,
                 post_bottleneck_size: int = 256,
                 output_size: int = 2,
                 manifold_size: int = 64,
                 learning_rate: float = 0.001):
        
        self.input_size = input_size
        self.pre_bottleneck_size = pre_bottleneck_size
        self.workspace_size = workspace_size
        self.post_bottleneck_size = post_bottleneck_size
        self.output_size = output_size
        self.manifold_size = manifold_size
        
        self._net = _lib.create_sgw_amn(
            input_size, pre_bottleneck_size, workspace_size,
            post_bottleneck_size, output_size, manifold_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create SGW-AMN network")
    
    def __del__(self):
        if hasattr(self, '_net') and self._net:
            _lib.destroy_sgw_amn(self._net)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        output = np.zeros(self.output_size, dtype=np.float32)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.sgw_amn_forward(self._net, x_ptr, out_ptr)
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
            verbose: int = 1, track_competition: bool = True) -> 'SparseGlobalWorkspace':
        """Train the network"""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        n_samples = len(X)
        
        sparsity_history = [] if track_competition else None
        entropy_history = [] if track_competition else None
        
        for epoch in range(epochs):
            self.reset_memory()
            epoch_loss = 0.0
            
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_sample = X[idx]
                y_sample = y[idx]
                
                x_ptr = x_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                y_ptr = y_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                loss = _lib.sgw_amn_train(self._net, x_ptr, y_ptr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_samples
            
            if track_competition:
                sparsity_history.append(self.workspace_sparsity)
                entropy_history.append(self.competition_entropy)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                      f"Sparsity: {self.workspace_sparsity:.2%} - "
                      f"Entropy: {self.competition_entropy:.4f}")
        
        if track_competition:
            self._sparsity_history = np.array(sparsity_history)
            self._entropy_history = np.array(entropy_history)
        
        return self
    
    def reset_memory(self):
        """Reset memory states"""
        _lib.sgw_amn_reset_memory(self._net)
    
    @property
    def workspace_sparsity(self) -> float:
        """Fraction of workspace slots that are active (0-1)"""
        return _lib.sgw_amn_get_workspace_sparsity(self._net)
    
    @property
    def competition_entropy(self) -> float:
        """Entropy of competitive routing (high = more competitive)"""
        return _lib.sgw_amn_get_competition_entropy(self._net)
    
    @property
    def bottleneck_rate(self) -> float:
        """Compression ratio (workspace_size / pre_bottleneck_size)"""
        return _lib.sgw_amn_get_bottleneck_rate(self._net)
    
    def print_info(self):
        """Print network information"""
        _lib.sgw_amn_print_info(self._net)
    
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
    
    def get_sparsity_history(self) -> np.ndarray:
        """Get history of workspace sparsity during training"""
        if hasattr(self, '_sparsity_history'):
            return self._sparsity_history
        else:
            raise RuntimeError("No history available. Train with track_competition=True")
    
    def get_entropy_history(self) -> np.ndarray:
        """Get history of competition entropy during training"""
        if hasattr(self, '_entropy_history'):
            return self._entropy_history
        else:
            raise RuntimeError("No history available. Train with track_competition=True")

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate SGW-AMN capabilities"""
    print("\n" + "="*70)
    print("SPARSE GLOBAL WORKSPACE DEMONSTRATION")
    print("Competitive Attention & Sparse Reasoning")
    print("="*70)
    
    np.random.seed(42)
    
    # Create network with aggressive compression
    net = SparseGlobalWorkspace(
        input_size=20,
        pre_bottleneck_size=128,
        workspace_size=8,  # Only 8 slots for 128 neurons to compete for!
        post_bottleneck_size=128,
        output_size=3,
        manifold_size=32,
        learning_rate=0.01
    )
    
    print("\nNetwork Architecture:")
    net.print_info()
    
    print(f"\nCompression Ratio: {net.bottleneck_rate:.4f}")
    print(f"  → 128 neurons competing for 8 workspace slots")
    print(f"  → Only {net.bottleneck_rate*100:.1f}% of information passes through!")
    
    # Create data
    n_samples = 200
    X = np.random.randn(n_samples, 20).astype(np.float32)
    y = np.random.randn(n_samples, 3).astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    # Train
    print("\nTraining with competition tracking...")
    print("(Watch how neurons compete for workspace access)\n")
    
    net.fit(X_train, y_train, epochs=50, verbose=1, track_competition=True)
    
    # Test
    print("\nTesting...")
    mse = net.score(X_test, y_test, metric='mse')
    r2 = net.score(X_test, y_test, metric='r2')
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R²:  {r2:.4f}")
    
    # Analyze competition
    print("\n" + "="*70)
    print("COMPETITIVE ROUTING ANALYSIS")
    print("="*70)
    
    sparsity_hist = net.get_sparsity_history()
    entropy_hist = net.get_entropy_history()
    
    print(f"\nWorkspace Sparsity Evolution:")
    print(f"  Initial: {sparsity_hist[0]:.2%} of workspace active")
    print(f"  Final:   {sparsity_hist[-1]:.2%} of workspace active")
    print(f"  Average: {sparsity_hist.mean():.2%}")
    
    print(f"\nCompetition Entropy Evolution:")
    print(f"  Initial: {entropy_hist[0]:.4f}")
    print(f"  Final:   {entropy_hist[-1]:.4f}")
    print(f"  (Higher entropy = more neurons competing)")
    
    # Demonstrate different input complexities
    print("\n" + "="*70)
    print("WORKSPACE UTILIZATION BY INPUT COMPLEXITY")
    print("="*70)
    
    # Simple input (low variance)
    simple_input = np.ones((1, 20), dtype=np.float32) * 0.1
    net.reset_memory()
    _ = net.predict(simple_input, reset_memory=False)
    simple_sparsity = net.workspace_sparsity
    simple_entropy = net.competition_entropy
    
    # Complex input (high variance)
    complex_input = np.random.randn(1, 20).astype(np.float32) * 3
    net.reset_memory()
    _ = net.predict(complex_input, reset_memory=False)
    complex_sparsity = net.workspace_sparsity
    complex_entropy = net.competition_entropy
    
    print(f"\nSimple Input:")
    print(f"  Workspace Sparsity: {simple_sparsity:.2%}")
    print(f"  Competition Entropy: {simple_entropy:.4f}")
    
    print(f"\nComplex Input:")
    print(f"  Workspace Sparsity: {complex_sparsity:.2%}")
    print(f"  Competition Entropy: {complex_entropy:.4f}")
    
    if complex_sparsity > simple_sparsity:
        print("\n  → Complex inputs use MORE workspace slots")
        print("  → Network adapts capacity to input complexity!")
    
    # Information bottleneck visualization
    print("\n" + "="*70)
    print("INFORMATION BOTTLENECK EFFECT")
    print("="*70)
    
    print(f"\nThe network forces attention by compression:")
    print(f"  Pre-bottleneck:  {net.pre_bottleneck_size} neurons (rich representation)")
    print(f"  Workspace:       {net.workspace_size} neurons (compressed essence)")
    print(f"  Post-bottleneck: {net.post_bottleneck_size} neurons (expanded back)")
    print(f"\nThis creates a 'consciousness-like' bottleneck where only")
    print(f"the most essential information survives the compression.")
    
    print("\n✓ Demonstration complete!")
    print("  - Neurons compete for limited workspace access")
    print("  - Compression forces extraction of essential features")
    print("  - Sparse representation enables efficient reasoning")
    print("  - Inspired by Global Workspace Theory of consciousness")

if __name__ == "__main__":
    demo()