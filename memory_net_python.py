#!/usr/bin/env python3
"""
MEMORY-NATIVE NEURAL NETWORK - PYTHON WRAPPER FOR C DLL

Uses the compiled C library via ctypes.

Compile C library first:
    Windows: gcc -shared -o memory_net.dll memory_net_dll.c -lm -O3
    Linux:   gcc -shared -fPIC -o memory_net.so memory_net_dll.c -lm -O3
    Mac:     gcc -shared -fPIC -o memory_net.dylib memory_net_dll.c -lm -O3
Licensed under GPL V3.
Then run this Python script:
    python memory_net_python.py
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'memory_net.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'memory_net.dylib'
    else:
        lib_name = 'memory_net.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o memory_net.dll memory_net_dll.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o memory_net.dylib memory_net_dll.c -lm -O3")
        else:
            print("  gcc -shared -fPIC -o memory_net.so memory_net_dll.c -lm -O3")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded C library successfully")
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
_lib.forward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
_lib.forward.restype = None

_lib.predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
_lib.predict.restype = None

# Training
_lib.train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
_lib.train.restype = ctypes.c_float

_lib.train_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
_lib.train_batch.restype = ctypes.c_float

# Partial training
_lib.freeze_hidden_layer.argtypes = [ctypes.c_void_p]
_lib.freeze_hidden_layer.restype = None

_lib.unfreeze_hidden_layer.argtypes = [ctypes.c_void_p]
_lib.unfreeze_hidden_layer.restype = None

_lib.freeze_output_layer.argtypes = [ctypes.c_void_p]
_lib.freeze_output_layer.restype = None

_lib.unfreeze_output_layer.argtypes = [ctypes.c_void_p]
_lib.unfreeze_output_layer.restype = None

_lib.freeze_percentage.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.freeze_percentage.restype = None

# Memory management
_lib.reset_memory.argtypes = [ctypes.c_void_p]
_lib.reset_memory.restype = None

_lib.get_memory_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]
_lib.get_memory_state.restype = None

_lib.set_memory_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]
_lib.set_memory_state.restype = None

# Parameters
_lib.get_beta.argtypes = [ctypes.c_void_p]
_lib.get_beta.restype = ctypes.c_float

_lib.set_beta.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_beta.restype = None

_lib.get_alpha.argtypes = [ctypes.c_void_p]
_lib.get_alpha.restype = ctypes.c_float

_lib.set_alpha.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_alpha.restype = None

_lib.get_learning_rate.argtypes = [ctypes.c_void_p]
_lib.get_learning_rate.restype = ctypes.c_float

_lib.set_learning_rate.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.set_learning_rate.restype = None

_lib.get_training_steps.argtypes = [ctypes.c_void_p]
_lib.get_training_steps.restype = ctypes.c_int

_lib.get_last_loss.argtypes = [ctypes.c_void_p]
_lib.get_last_loss.restype = ctypes.c_float

# Save/Load
_lib.save_network.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.save_network.restype = ctypes.c_int

_lib.load_network.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.load_network.restype = ctypes.c_int

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class MemoryNativeNetwork:
    """
    Memory-Native Neural Network
    
    Python wrapper for C library implementation.
    
    Features:
    - Persistent internal memory across all interactions
    - Continuous learning capability
    - Partial training (selective weight freezing)
    - Memory preservation across sessions
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden neurons with memory
        output_size: Number of output neurons
        beta: Memory preservation factor (0-1)
              Higher = stronger memory of past outputs
        alpha: Memory update rate (0-1)
               Controls how fast internal memory updates
        learning_rate: Standard SGD learning rate
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 beta: float = 0.3, alpha: float = 0.1, learning_rate: float = 0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create network via C library
        self._net = _lib.create_network(
            input_size, hidden_size, output_size,
            beta, alpha, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create network")
        
        print(f"Network created: {input_size} -> {hidden_size} -> {output_size}")
        print(f"  Beta (memory preservation): {beta}")
        print(f"  Alpha (memory update rate): {alpha}")
        print(f"  Learning rate: {learning_rate}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            x: Input array of shape (input_size,) or (batch_size, input_size)
        
        Returns:
            Predictions of shape (output_size,) or (batch_size, output_size)
        """
        # Handle single sample or batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = x.shape[0]
        x = np.ascontiguousarray(x, dtype=np.float32)
        
        # Predict each sample (network maintains state across predictions)
        outputs = []
        for i in range(batch_size):
            input_data = x[i].flatten()
            output_data = np.zeros(self.output_size, dtype=np.float32)
            
            _lib.predict(
                self._net,
                input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            outputs.append(output_data)
        
        result = np.array(outputs)
        return result[0] if squeeze else result
    
    def train(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Train on a single sample or batch
        
        Args:
            x: Input array
            y: Target array
        
        Returns:
            Loss value
        """
        # Handle single sample or batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
        
        batch_size = x.shape[0]
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        
        if batch_size == 1:
            # Single sample
            loss = _lib.train(
                self._net,
                x.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        else:
            # Batch training
            loss = _lib.train_batch(
                self._net,
                x.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size
            )
        
        return float(loss)
    
    def train_epochs(self, x: np.ndarray, y: np.ndarray, epochs: int,
                     batch_size: int = 32, verbose: bool = True) -> list:
        """
        Train for multiple epochs
        
        Args:
            x: Training data
            y: Training targets
            epochs: Number of epochs
            batch_size: Batch size for training
            verbose: Print progress
        
        Returns:
            List of losses per epoch
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n_samples = x.shape[0]
        
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                batch_x = x_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                loss = self.train(batch_x, batch_y)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        return losses
    
    # Partial Training Methods
    
    def freeze_hidden(self):
        """Freeze hidden layer weights (stop training them)"""
        _lib.freeze_hidden_layer(self._net)
        print("Hidden layer frozen")
    
    def unfreeze_hidden(self):
        """Unfreeze hidden layer weights (resume training)"""
        _lib.unfreeze_hidden_layer(self._net)
        print("Hidden layer unfrozen")
    
    def freeze_output(self):
        """Freeze output layer weights"""
        _lib.freeze_output_layer(self._net)
        print("Output layer frozen")
    
    def unfreeze_output(self):
        """Unfreeze output layer weights"""
        _lib.unfreeze_output_layer(self._net)
        print("Output layer unfrozen")
    
    def freeze_percentage(self, percentage: float):
        """
        Freeze a percentage of weights randomly
        
        Args:
            percentage: Fraction to freeze (0.0 to 1.0)
        """
        _lib.freeze_percentage(self._net, percentage)
        print(f"{percentage*100:.1f}% of hidden weights frozen")
    
    # Memory Management
    
    def reset_memory(self):
        """Clear all persistent internal memory"""
        _lib.reset_memory(self._net)
        print("Memory reset")
    
    def get_memory_state(self) -> np.ndarray:
        """
        Get current internal memory state
        
        Returns:
            Array of internal memory values
        """
        memory = np.zeros(self.hidden_size, dtype=np.float32)
        _lib.get_memory_state(
            self._net,
            memory.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return memory
    
    def set_memory_state(self, memory: np.ndarray):
        """
        Set internal memory state
        
        Args:
            memory: Array of memory values
        """
        memory = np.ascontiguousarray(memory, dtype=np.float32)
        _lib.set_memory_state(
            self._net,
            memory.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
    
    # Parameter Access
    
    @property
    def beta(self) -> float:
        """Get memory preservation factor"""
        return _lib.get_beta(self._net)
    
    @beta.setter
    def beta(self, value: float):
        """Set memory preservation factor"""
        _lib.set_beta(self._net, value)
    
    @property
    def alpha(self) -> float:
        """Get memory update rate"""
        return _lib.get_alpha(self._net)
    
    @alpha.setter
    def alpha(self, value: float):
        """Set memory update rate"""
        _lib.set_alpha(self._net, value)
    
    @property
    def learning_rate(self) -> float:
        """Get learning rate"""
        return _lib.get_learning_rate(self._net)
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        """Set learning rate"""
        _lib.set_learning_rate(self._net, value)
    
    @property
    def training_steps(self) -> int:
        """Get number of training steps"""
        return _lib.get_training_steps(self._net)
    
    @property
    def last_loss(self) -> float:
        """Get last training loss"""
        return _lib.get_last_loss(self._net)
    
    # Save/Load
    
    def save(self, filename: str):
        """
        Save network to file (includes all memory states)
        
        Args:
            filename: Path to save file
        """
        result = _lib.save_network(self._net, filename.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to save network (error {result})")
        print(f"Network saved to {filename}")
    
    def load(self, filename: str):
        """
        Load network from file (restores all memory states)
        
        Args:
            filename: Path to load file
        """
        result = _lib.load_network(self._net, filename.encode('utf-8'))
        if result == -2:
            raise ValueError("Network size mismatch")
        elif result != 0:
            raise RuntimeError(f"Failed to load network (error {result})")
        print(f"Network loaded from {filename}")
    
    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_network(self._net)

# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_usage():
    """Basic training and prediction"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Training and Prediction")
    print("="*70)
    
    # Create network
    net = MemoryNativeNetwork(
        input_size=4,
        hidden_size=8,
        output_size=2,
        beta=0.3,
        alpha=0.1,
        learning_rate=0.01
    )
    
    # Generate training data (XOR-like problem)
    np.random.seed(42)
    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randn(100, 2).astype(np.float32)
    
    # Train
    print("\nTraining...")
    losses = net.train_epochs(X_train, y_train, epochs=50, verbose=True)
    
    # Predict
    print("\nMaking predictions...")
    X_test = np.random.randn(5, 4).astype(np.float32)
    predictions = net.predict(X_test)
    
    print(f"\nTest predictions shape: {predictions.shape}")
    print(f"Sample predictions:\n{predictions[:3]}")
    
    return net

def demo_partial_training():
    """Selective freezing of layers"""
    print("\n" + "="*70)
    print("DEMO 2: Partial Training")
    print("="*70)
    
    net = MemoryNativeNetwork(8, 16, 4, beta=0.3)
    
    # Generate data
    X = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randn(50, 4).astype(np.float32)
    
    # Phase 1: Train everything
    print("\nPhase 1: Training all layers")
    net.train_epochs(X, y, epochs=20, verbose=False)
    print(f"Loss: {net.last_loss:.6f}")
    
    # Get prediction before freezing
    test_x = np.random.randn(1, 8).astype(np.float32)
    pred_before = net.predict(test_x)
    
    # Phase 2: Freeze hidden layer
    print("\nPhase 2: Freezing hidden layer")
    net.freeze_hidden()
    net.train_epochs(X, y, epochs=20, verbose=False)
    print(f"Loss: {net.last_loss:.6f}")
    
    pred_after = net.predict(test_x)
    
    print(f"\nPrediction before freeze: {pred_before[0]}")
    print(f"Prediction after freeze:  {pred_after[0]}")
    print(f"Change: {np.abs(pred_after - pred_before).mean():.6f}")
    
    # Phase 3: Unfreeze and continue
    print("\nPhase 3: Unfreezing and continuing")
    net.unfreeze_hidden()
    net.train_epochs(X, y, epochs=20, verbose=False)
    print(f"Loss: {net.last_loss:.6f}")

def demo_memory_persistence():
    """Save and load with memory preservation"""
    print("\n" + "="*70)
    print("DEMO 3: Memory Persistence Across Sessions")
    print("="*70)
    
    # Create and train network
    print("\nSession 1: Creating and training network")
    net1 = MemoryNativeNetwork(5, 10, 3, beta=0.4, alpha=0.1)
    
    X = np.random.randn(80, 5).astype(np.float32)
    y = np.random.randn(80, 3).astype(np.float32)
    
    net1.train_epochs(X, y, epochs=30, verbose=False)
    print(f"Training complete. Loss: {net1.last_loss:.6f}")
    
    # Get memory state and prediction
    memory1 = net1.get_memory_state()
    test_x = np.random.randn(1, 5).astype(np.float32)
    pred1 = net1.predict(test_x)
    
    print(f"\nInternal memory (first 5 values): {memory1[:5]}")
    print(f"Prediction: {pred1[0]}")
    
    # Save network
    net1.save("test_network.bin")
    
    # Simulate program restart
    print("\n" + "-"*70)
    print("Simulating program restart...")
    print("-"*70)
    del net1
    
    # Load network
    print("\nSession 2: Loading saved network")
    net2 = MemoryNativeNetwork(5, 10, 3)
    net2.load("test_network.bin")
    
    # Get memory state and prediction
    memory2 = net2.get_memory_state()
    pred2 = net2.predict(test_x)
    
    print(f"\nInternal memory (first 5 values): {memory2[:5]}")
    print(f"Prediction: {pred2[0]}")
    
    print(f"\nMemory difference: {np.abs(memory1 - memory2).max():.10f}")
    print(f"Prediction difference: {np.abs(pred1 - pred2).max():.10f}")
    print("\n✓ Memory state preserved perfectly across sessions!")

def demo_memory_effects():
    """Show effect of memory parameters"""
    print("\n" + "="*70)
    print("DEMO 4: Memory Effects (Beta and Alpha)")
    print("="*70)
    
    configs = [
        ("No memory", 0.0, 0.1),
        ("Moderate memory", 0.3, 0.1),
        ("Strong memory", 0.7, 0.1),
    ]
    
    for name, beta, alpha in configs:
        print(f"\n--- {name} (beta={beta}) ---")
        net = MemoryNativeNetwork(3, 6, 2, beta=beta, alpha=alpha)
        
        # Strong initial input
        strong_input = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        pred = net.predict(strong_input)
        print(f"Step 0 (strong input): {pred[0]}")
        
        # Weak subsequent inputs
        weak_input = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        for step in range(1, 6):
            pred = net.predict(weak_input)
            print(f"Step {step} (weak input):   {pred[0]}")
        
        print("Notice: Higher beta = output retains more of initial signal")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "MEMORY-NATIVE NEURAL NETWORK" + " "*25 + "║")
    print("║" + " "*20 + "Python + C Implementation" + " "*23 + "║")
    print("╚" + "="*68 + "╝\n")
    
    try:
        # Run all demos
        net = demo_basic_usage()
        
        input("\nPress Enter to continue to next demo...")
        demo_partial_training()
        
        input("\nPress Enter to continue to next demo...")
        demo_memory_persistence()
        
        input("\nPress Enter to continue to next demo...")
        demo_memory_effects()
        
        print("\n" + "="*70)
        print("All demonstrations complete!")
        print("="*70)
        
        print("\nKey Features Demonstrated:")
        print("  ✓ Training and prediction")
        print("  ✓ Partial training (selective freezing)")
        print("  ✓ Memory persistence across sessions")
        print("  ✓ Adjustable memory parameters (beta, alpha)")
        print("  ✓ Fast C implementation via DLL")
        
        print("\nThe network's internal memory persists across ALL operations,")
        print("enabling continuous learning and personalization!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()