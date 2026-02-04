#!/usr/bin/env python3
"""
NEURAL DIFFERENTIAL MANIFOLDS WITH MOMENTUM (NDM-M) - PYTHON API

Advanced architecture with momentum-damped continuous weight evolution:
V(t+1) = β * V(t) + α * dW/dt
W(t+1) = W(t) + V(t+1)

Key Innovation: Momentum provides smoother, more stable neuroplasticity
with better convergence properties and reduced oscillations

Compile C library first:
    Windows: gcc -shared -o ndm_momentum.dll ndm_momentum.c -lm -O3
    Linux:   gcc -shared -fPIC -o ndm_momentum.so ndm_momentum.c -lm -O3
    Mac:     gcc -shared -fPIC -o ndm_momentum.dylib ndm_momentum.c -lm -O3

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
        lib_name = 'ndm_momentum.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'ndm_momentum.dylib'
    else:
        lib_name = 'ndm_momentum.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o ndm_momentum.dll ndm_momentum.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o ndm_momentum.dylib ndm_momentum.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o ndm_momentum.so ndm_momentum.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

try:
    _lib = load_library()
    print(f"✓ Loaded NDM-Momentum C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

_lib.create_ndm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_ndm.restype = ctypes.c_void_p
_lib.destroy_ndm.argtypes = [ctypes.c_void_p]
_lib.destroy_ndm.restype = None

_lib.ndm_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.ndm_forward.restype = None

_lib.ndm_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.ndm_train.restype = ctypes.c_float

_lib.ndm_reset_memory.argtypes = [ctypes.c_void_p]
_lib.ndm_reset_memory.restype = None

_lib.ndm_get_avg_weight_velocity.argtypes = [ctypes.c_void_p]
_lib.ndm_get_avg_weight_velocity.restype = ctypes.c_float

_lib.ndm_get_avg_plasticity.argtypes = [ctypes.c_void_p]
_lib.ndm_get_avg_plasticity.restype = ctypes.c_float

_lib.ndm_get_avg_manifold_energy.argtypes = [ctypes.c_void_p]
_lib.ndm_get_avg_manifold_energy.restype = ctypes.c_float

_lib.ndm_get_momentum.argtypes = [ctypes.c_void_p]
_lib.ndm_get_momentum.restype = ctypes.c_float

_lib.ndm_set_momentum.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.ndm_set_momentum.restype = None

_lib.ndm_print_info.argtypes = [ctypes.c_void_p]
_lib.ndm_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class NeuralDifferentialManifold:
    """
    Neural Differential Manifolds with Momentum (NDM-M)
    
    Network where weights evolve as differential equations with momentum:
    V(t+1) = β * V(t) + α * dW/dt
    W(t+1) = W(t) + V(t+1)
    
    This creates momentum-damped neuroplasticity - the network physically 
    rewires its connections smoothly with reduced oscillations and better
    convergence properties.
    
    Features:
    - Continuous weight evolution via ODEs with momentum damping
    - Hebbian learning ("neurons that fire together wire together")
    - Adaptive plasticity that increases with prediction errors
    - Associative memory manifold for long-term patterns
    - Smoother learning dynamics compared to standard NDM
    
    Parameters
    ----------
    input_size : int
        Dimension of input vectors
    hidden_size : int
        Number of hidden neurons
    output_size : int
        Dimension of output vectors
    manifold_size : int, default=64
        Size of the associative memory matrix
    learning_rate : float, default=0.001
        Learning rate for supervised training
    momentum : float, default=0.9
        Momentum coefficient (β) for velocity updates
    
    Examples
    --------
    >>> net = NeuralDifferentialManifold(input_size=10, hidden_size=32, output_size=2)
    >>> X = np.random.randn(100, 10).astype(np.float32)
    >>> y = np.random.randn(100, 2).astype(np.float32)
    >>> net.fit(X, y, epochs=50)
    >>> print(f"Weight velocity: {net.avg_weight_velocity:.6f}")
    >>> print(f"Plasticity: {net.avg_plasticity:.4f}")
    >>> print(f"Momentum: {net.momentum:.4f}")
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 manifold_size: int = 64, learning_rate: float = 0.001,
                 momentum: float = 0.9):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.manifold_size = manifold_size
        
        self._net = _lib.create_ndm(
            input_size, hidden_size, output_size, manifold_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create NDM network")
        
        # Set momentum if provided
        if momentum != 0.9:  # Only set if different from default
            self.momentum = momentum
    
    def __del__(self):
        if hasattr(self, '_net') and self._net:
            _lib.destroy_ndm(self._net)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with momentum-damped weight evolution"""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        output = np.zeros(self.output_size, dtype=np.float32)
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.ndm_forward(self._net, x_ptr, out_ptr)
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
            verbose: int = 1, track_plasticity: bool = True) -> 'NeuralDifferentialManifold':
        """
        Train the network with momentum-damped weight evolution
        
        The network adapts in three ways:
        1. Supervised learning on output weights (with momentum)
        2. Unsupervised weight evolution via Hebbian plasticity (with momentum)
        3. Adaptive plasticity that responds to prediction errors
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        n_samples = len(X)
        
        plasticity_history = [] if track_plasticity else None
        velocity_history = [] if track_plasticity else None
        
        for epoch in range(epochs):
            self.reset_memory()
            epoch_loss = 0.0
            
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_sample = X[idx]
                y_sample = y[idx]
                
                x_ptr = x_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                y_ptr = y_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                loss = _lib.ndm_train(self._net, x_ptr, y_ptr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_samples
            
            if track_plasticity:
                plasticity_history.append(self.avg_plasticity)
                velocity_history.append(self.avg_weight_velocity)
            
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - "
                      f"Plasticity: {self.avg_plasticity:.4f} - "
                      f"Velocity: {self.avg_weight_velocity:.6f} - "
                      f"Momentum: {self.momentum:.3f}")
        
        if track_plasticity:
            self._plasticity_history = np.array(plasticity_history)
            self._velocity_history = np.array(velocity_history)
        
        return self
    
    def reset_memory(self):
        """Reset memory states (but not learned weights or momentum)"""
        _lib.ndm_reset_memory(self._net)
    
    @property
    def avg_weight_velocity(self) -> float:
        """Average rate of weight change (includes momentum effects)"""
        return _lib.ndm_get_avg_weight_velocity(self._net)
    
    @property
    def avg_plasticity(self) -> float:
        """Average plasticity across network (0=rigid, 1=fluid)"""
        return _lib.ndm_get_avg_plasticity(self._net)
    
    @property
    def avg_manifold_energy(self) -> float:
        """Energy in associative memory manifold"""
        return _lib.ndm_get_avg_manifold_energy(self._net)
    
    @property
    def momentum(self) -> float:
        """Current momentum coefficient (β)"""
        return _lib.ndm_get_momentum(self._net)
    
    @momentum.setter
    def momentum(self, value: float):
        """Set momentum coefficient (β), must be in [0, 0.999]"""
        if not 0.0 <= value < 1.0:
            raise ValueError("Momentum must be in [0, 1)")
        _lib.ndm_set_momentum(self._net, value)
    
    def print_info(self):
        """Print network information"""
        _lib.ndm_print_info(self._net)
    
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
    
    def get_plasticity_history(self) -> np.ndarray:
        """Get history of plasticity during training"""
        if hasattr(self, '_plasticity_history'):
            return self._plasticity_history
        else:
            raise RuntimeError("No history available. Train with track_plasticity=True")
    
    def get_velocity_history(self) -> np.ndarray:
        """Get history of weight velocity during training"""
        if hasattr(self, '_velocity_history'):
            return self._velocity_history
        else:
            raise RuntimeError("No history available. Train with track_plasticity=True")

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate NDM-Momentum capabilities"""
    print("\n" + "="*70)
    print("NEURAL DIFFERENTIAL MANIFOLDS WITH MOMENTUM DEMONSTRATION")
    print("Momentum-Damped Continuous Weight Evolution")
    print("="*70)
    
    np.random.seed(42)
    
    # Create network
    net = NeuralDifferentialManifold(
        input_size=10,
        hidden_size=32,
        output_size=3,
        manifold_size=32,
        learning_rate=0.01,
        momentum=0.9
    )
    
    print("\nNetwork Architecture:")
    net.print_info()
    
    # Create data
    n_samples = 200
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = np.random.randn(n_samples, 3).astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    # Train
    print("\nTraining with momentum-damped neuroplasticity...")
    print("(Watch how momentum smooths weight evolution)\n")
    
    net.fit(X_train, y_train, epochs=50, verbose=1, track_plasticity=True)
    
    # Test
    print("\nTesting...")
    mse = net.score(X_test, y_test, metric='mse')
    r2 = net.score(X_test, y_test, metric='r2')
    
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test R²:  {r2:.4f}")
    
    # Analyze plasticity evolution
    print("\n" + "="*70)
    print("MOMENTUM-DAMPED NEUROPLASTICITY ANALYSIS")
    print("="*70)
    
    plasticity_hist = net.get_plasticity_history()
    velocity_hist = net.get_velocity_history()
    
    print(f"\nPlasticity Evolution:")
    print(f"  Initial: {plasticity_hist[0]:.4f} (network starts fluid)")
    print(f"  Final:   {plasticity_hist[-1]:.4f} (network becomes more rigid)")
    print(f"  Decay:   {(plasticity_hist[0] - plasticity_hist[-1]) / plasticity_hist[0] * 100:.1f}%")
    
    print(f"\nWeight Velocity Evolution (with momentum):")
    print(f"  Initial: {velocity_hist[0]:.6f}")
    print(f"  Final:   {velocity_hist[-1]:.6f}")
    print(f"  Peak:    {velocity_hist.max():.6f} at epoch {velocity_hist.argmax() + 1}")
    print(f"  Note: Momentum coefficient β = {net.momentum:.3f}")
    
    print(f"\nCurrent State:")
    print(f"  Manifold Energy: {net.avg_manifold_energy:.6f}")
    print(f"  Weight Velocity: {net.avg_weight_velocity:.6f}")
    print(f"  Plasticity:      {net.avg_plasticity:.4f}")
    print(f"  Momentum (β):    {net.momentum:.3f}")
    
    # Demonstrate momentum effects
    print("\n" + "="*70)
    print("MOMENTUM COMPARISON")
    print("="*70)
    
    print("\nComparing different momentum values...")
    
    momentum_values = [0.0, 0.5, 0.9, 0.95]
    results = []
    
    for mom in momentum_values:
        net_test = NeuralDifferentialManifold(
            input_size=10,
            hidden_size=16,
            output_size=3,
            manifold_size=16,
            learning_rate=0.01,
            momentum=mom
        )
        
        # Train briefly
        net_test.fit(X_train[:50], y_train[:50], epochs=20, verbose=0, track_plasticity=True)
        
        # Get final velocity
        final_velocity = net_test.avg_weight_velocity
        velocity_std = np.std(net_test.get_velocity_history())
        
        results.append((mom, final_velocity, velocity_std))
        
        print(f"  β={mom:.2f}: Velocity={final_velocity:.6f}, "
              f"Std={velocity_std:.6f} {'(smoother)' if velocity_std < 0.01 else ''}")
    
    print("\n  Higher momentum → Smoother weight evolution (lower std)")
    print("  Lower momentum  → More responsive but noisier updates")
    
    # Demonstrate adaptive momentum
    print("\n" + "="*70)
    print("ADAPTIVE MOMENTUM DEMONSTRATION")
    print("="*70)
    
    print("\nYou can adjust momentum dynamically during training:")
    
    net_adaptive = NeuralDifferentialManifold(
        input_size=10,
        hidden_size=16,
        output_size=3,
        learning_rate=0.01,
        momentum=0.5  # Start with lower momentum
    )
    
    print(f"\n  Starting momentum: {net_adaptive.momentum:.2f}")
    
    # Train with lower momentum initially
    net_adaptive.fit(X_train[:50], y_train[:50], epochs=10, verbose=0)
    
    # Increase momentum for fine-tuning
    net_adaptive.momentum = 0.95
    print(f"  Increased to:      {net_adaptive.momentum:.2f} (for fine-tuning)")
    
    net_adaptive.fit(X_train[:50], y_train[:50], epochs=10, verbose=0)
    
    print("\n  Strategy: Low momentum early (explore) → High momentum late (refine)")
    
    print("\n" + "="*70)
    print("✓ Demonstration complete!")
    print("="*70)
    print("\nKey Advantages of Momentum:")
    print("  ✓ Smoother weight evolution (reduced oscillations)")
    print("  ✓ Better convergence in complex loss landscapes")
    print("  ✓ Helps escape shallow local minima")
    print("  ✓ Reduces sensitivity to learning rate")
    print("  ✓ More stable neuroplasticity dynamics")
    print("\nMomentum Formula:")
    print("  V(t+1) = β * V(t) + α * dW/dt")
    print("  W(t+1) = W(t) + V(t+1)")
    print("\n  where β = momentum coefficient (typically 0.9)")
    print("        α = learning rate / plasticity")

def compare_with_without_momentum():
    """Side-by-side comparison of NDM with and without momentum"""
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON: WITH vs WITHOUT MOMENTUM")
    print("="*70)
    
    np.random.seed(42)
    
    # Create data
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100, 3).astype(np.float32)
    
    # Train without momentum
    print("\n[1] Training WITHOUT momentum (β=0.0)...")
    net_no_momentum = NeuralDifferentialManifold(
        input_size=10, hidden_size=32, output_size=3,
        learning_rate=0.01, momentum=0.0
    )
    net_no_momentum.fit(X, y, epochs=30, verbose=0, track_plasticity=True)
    
    # Train with momentum
    print("[2] Training WITH momentum (β=0.9)...")
    net_momentum = NeuralDifferentialManifold(
        input_size=10, hidden_size=32, output_size=3,
        learning_rate=0.01, momentum=0.9
    )
    net_momentum.fit(X, y, epochs=30, verbose=0, track_plasticity=True)
    
    # Compare
    print("\nResults:")
    print(f"\n  WITHOUT Momentum (β=0.0):")
    print(f"    Final Loss:       {net_no_momentum.score(X, y):.6f}")
    print(f"    Velocity Std:     {np.std(net_no_momentum.get_velocity_history()):.6f}")
    print(f"    Final Velocity:   {net_no_momentum.avg_weight_velocity:.6f}")
    
    print(f"\n  WITH Momentum (β=0.9):")
    print(f"    Final Loss:       {net_momentum.score(X, y):.6f}")
    print(f"    Velocity Std:     {np.std(net_momentum.get_velocity_history()):.6f} ← Smoother!")
    print(f"    Final Velocity:   {net_momentum.avg_weight_velocity:.6f}")
    
    print("\n  → Momentum provides smoother, more stable learning dynamics")

if __name__ == "__main__":
    demo()
    print("\n")
    compare_with_without_momentum()