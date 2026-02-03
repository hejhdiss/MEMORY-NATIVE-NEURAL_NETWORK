# Memory-Native Neural Networks API

A clean, unified Python API for memory-native neural network architectures with persistent internal states.

## 🧠 Overview

This API provides a simple interface to three powerful memory-native neural network models:

### AMRC (Adaptive Memory Recurrent Cell)
Basic memory-native architecture with:
- **Memory-preserving activation** (β parameter) - retains past output information
- **Stateful neurons** (α parameter) - internal memory that evolves over time
- **Partial training** - selective layer freezing for transfer learning
- **High-performance C backend** for fast computation

### PMRC (Persistent Memory Recurrent Cell)
Advanced architecture extending AMRC with:
- **Learnable memory gates** - network learns what to remember
- **Output layer memory** - memory at every layer
- **Advanced partial training** - freeze by percentage, magnitude, or specific neurons
- **Multiple serialization formats** - custom binary, joblib, or pickle

### AMN (Adaptive Memory Network) **NEW**
State-of-the-art architecture combining three cutting-edge mechanisms:
- **Liquid Constant (LC) neurons** - dynamic time constants that adapt based on input importance
- **Linear Recurrent Units (LRU)** - efficient parallel-friendly recurrent processing
- **Associative Memory Manifolds (AMM)** - global memory whiteboard for long-range dependencies
- **Advanced diagnostics** - track manifold energy, LRU magnitude, and LC timescales

## 📦 Installation

### Prerequisites

1. **Compile the C libraries** (required for all models):

```bash
# For AMRC
gcc -shared -o memory_net.dll memory_net_dll.c -lm -O3        # Windows
gcc -shared -fPIC -o memory_net.so memory_net_dll.c -lm -O3   # Linux
gcc -shared -fPIC -o memory_net.dylib memory_net_dll.c -lm -O3 # macOS

# For PMRC
gcc -shared -o memory_net_extended.dll memory_net_extended.c -lm -O3        # Windows
gcc -shared -fPIC -o memory_net_extended.so memory_net_extended.c -lm -O3   # Linux
gcc -shared -fPIC -o memory_net_extended.dylib memory_net_extended.c -lm -O3 # macOS

# For AMN (requires OpenMP for parallel processing)
gcc -shared -o amn.dll amn.c -lm -O3 -fopenmp        # Windows
gcc -shared -fPIC -o amn.so amn.c -lm -O3 -fopenmp   # Linux
gcc -shared -fPIC -o amn.dylib amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp # macOS
```

2. **Install Python dependencies**:

```bash
pip install numpy
pip install joblib  # Optional, for PMRC serialization
```

### Import the API

```python
from api import AMRC, PMRC, AMN, create_model
```

## 🚀 Quick Start

### Using AMRC (Simple Memory Model)

```python
import numpy as np
from api import AMRC

# Create model
model = AMRC(
    input_size=10,
    hidden_size=20,
    output_size=5,
    beta=0.3,        # Memory preservation strength
    alpha=0.1,       # Memory update rate
    learning_rate=0.01
)

# Train
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randn(100, 5).astype(np.float32)
model.fit(X_train, y_train, epochs=50, verbose=1)

# Predict
X_test = np.random.randn(20, 10).astype(np.float32)
predictions = model.predict(X_test)

# Save/Load
model.save('my_model.bin')
model.load('my_model.bin')
```

### Using PMRC (Advanced Learnable Memory)

```python
from api import PMRC

# Create model with learnable gates
model = PMRC(
    input_size=10,
    hidden_size=20,
    output_size=5,
    use_learnable_gates=True,  # Network learns what to remember
    use_output_memory=True,    # Memory in output layer too
    learning_rate=0.01
)

# Train
model.fit(X_train, y_train, epochs=50, verbose=1)

# Check what the network learned to remember
print(f"Average gate value: {model.avg_gate_value:.4f}")
print(f"Gate states: {model.get_gate_state()[:5]}")

# Advanced partial training
model.freeze_hidden_percentage(0.5)  # Freeze 50% of hidden neurons
model.freeze_memory_gates()          # Fix memory dynamics
model.fit(X_new, y_new, epochs=20)

# Save with different formats
model.save('model.bin', method='custom')   # Fast binary format
model.save('model.joblib', method='joblib') # Joblib format
model.save('model.pkl', method='pickle')    # Pickle format
```

### Using AMN (Adaptive Memory Network) **NEW**

```python
from api import AMN

# Create AMN with adaptive memory mechanisms
model = AMN(
    input_size=10,
    hidden_size=32,
    output_size=5,
    memory_manifold_size=64,  # Size of associative memory
    learning_rate=0.001,
    dt=0.1,                   # Time step for liquid constants
    memory_decay=0.995        # Decay rate for manifold memory
)

# Train
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Monitor adaptive mechanisms
print(f"Manifold Energy: {model.avg_manifold_energy:.4f}")
print(f"LC Timescale: {model.avg_lc_timescale:.4f}")
print(f"LRU Magnitude: {model.avg_lru_magnitude:.4f}")

# Predict
predictions = model.predict(X_test, reset_memory=False)

# Save/Load
model.save('amn_model.bin')
loaded_model = AMN.load('amn_model.bin')
```

### Using the Factory Function

```python
from api import create_model

# Create any model type dynamically
model = create_model('amrc', input_size=10, hidden_size=20, output_size=5)
# or
model = create_model('pmrc', input_size=10, hidden_size=20, output_size=5)
# or
model = create_model('amn', input_size=10, hidden_size=32, output_size=5)
```

## 📖 API Reference

### Common Methods (All Models: AMRC, PMRC, AMN)

#### `fit(X, y, epochs=100, batch_size=None, verbose=0, **kwargs)`
Train the model on data.

**Parameters:**
- `X` (np.ndarray): Training input, shape (n_samples, input_size)
- `y` (np.ndarray): Training targets, shape (n_samples, output_size)
- `epochs` (int): Number of training epochs
- `batch_size` (int, optional): Batch size for training (required for AMN, default=32)
- `verbose` (int): Verbosity level (0=silent, 1=progress, 2=detailed)
- `validation_split` (float): Fraction for validation (AMRC/PMRC only)
- `reset_memory` (bool): Reset memory before training (default: False for AMRC/PMRC, True for AMN)

**Returns:** `self` (for method chaining)

#### `predict(X, reset_memory=False)`
Make predictions.

**Parameters:**
- `X` (np.ndarray): Input data, shape (n_samples, input_size)
- `reset_memory` (bool): Whether to reset memory before prediction (AMN only)

**Returns:** Predictions, shape (n_samples, output_size)

#### `score(X, y, metric='r2')`
Compute performance score on test data.

**Parameters:**
- `X` (np.ndarray): Test input data
- `y` (np.ndarray): True target values
- `metric` (str): Metric to use - 'r2' (default), 'mse', 'mae' (AMN supports all, AMRC/PMRC use R²)

**Returns:** Performance score (float)

#### `reset_memory()`
Reset internal memory state to zero.

#### `get_memory_state()`
Get current internal memory state.

**Returns:** Memory state vector (np.ndarray)

**Note:** For AMN, this returns the hidden state (LRU state). Use `get_manifold_state()` for manifold memory.

#### `set_memory_state(memory)`
Set internal memory state.

**Parameters:**
- `memory` (np.ndarray): Memory state to set

**Note:** For AMN, this sets the hidden state. The manifold memory is managed separately.

#### `save(filepath, method='custom')`
Save model to file.

**Parameters:**
- `filepath` (str): Path to save file
- `method` (str): 'custom' (AMRC/AMN), 'custom'/'joblib'/'pickle' (PMRC)

#### `load(filepath, method='custom')`
Load model from file.

**Parameters:**
- `filepath` (str): Path to load from
- `method` (str): Serialization method used

**Note:** AMN uses a class method: `model = AMN.load('path.bin')`

### AMRC-Specific Methods

#### `freeze_hidden_layer()` / `unfreeze_hidden_layer()`
Freeze/unfreeze hidden layer weights during training.

#### `freeze_output_layer()` / `unfreeze_output_layer()`
Freeze/unfreeze output layer weights during training.

### PMRC-Specific Methods

#### `get_gate_state()`
Get learnable gate states (requires `use_learnable_gates=True`).

**Returns:** Gate state vector (np.ndarray)

#### `freeze_memory_gates()` / `unfreeze_memory_gates()`
Freeze/unfreeze learnable memory gate parameters.

#### `freeze_hidden_percentage(percentage)`
Freeze a percentage of hidden neurons.

**Parameters:**
- `percentage` (float): Percentage to freeze (0.0 to 1.0)

#### `freeze_output_percentage(percentage)`
Freeze a percentage of output neurons.

#### `freeze_by_magnitude(threshold, freeze_large=True)`
Freeze weights based on magnitude.

**Parameters:**
- `threshold` (float): Magnitude threshold
- `freeze_large` (bool): If True, freeze large weights; else freeze small weights

### AMN-Specific Methods

#### `get_manifold_state()`
Get the current associative memory manifold state.

**Returns:** Manifold matrix (np.ndarray), shape (memory_manifold_size, hidden_size)

#### `reset_manifold()`
Reset only the associative memory manifold to zero (keeps LRU/LC state).

#### `print_info()`
Print detailed network information and statistics.

### AMN-Specific Properties

#### `avg_manifold_energy`
Average energy in the associative memory manifold (float, read-only).

#### `avg_lru_magnitude`
Average magnitude of LRU states (float, read-only).

#### `avg_lc_timescale`
Average time constant of Liquid Constant neurons (float, read-only).

#### `dt`
Time step for Liquid Constant dynamics (float, read-write).

#### `memory_decay`
Decay rate for associative memory manifold (float, read-write, range: 0.99-0.9999).

## 🎯 Key Parameters

### Memory Parameters

| Parameter | Range | Models | Description | When to Use |
|-----------|-------|--------|-------------|-------------|
| `beta` | 0.0-1.0 | AMRC, PMRC | Memory preservation strength | Higher = stronger retention of past outputs |
| `alpha` | 0.0-1.0 | AMRC, PMRC | Memory update rate | Higher = faster memory updates |
| `output_beta` | 0.0-1.0 | PMRC | Output layer memory | Enable output-level memory preservation |
| `memory_manifold_size` | 16-512 | AMN | Size of associative memory | Larger = more memory capacity, slower |
| `dt` | 0.01-1.0 | AMN | Time step for LC dynamics | Smaller = finer temporal resolution |
| `memory_decay` | 0.99-0.9999 | AMN | Manifold decay rate | Higher = longer memory retention |

### Architecture Flags

| Flag | Default | Models | Description |
|------|---------|--------|-------------|
| `use_recurrent` | True | PMRC | Enable recurrent connections |
| `use_learnable_gates` | False | PMRC | Let network learn memory dynamics |
| `use_output_memory` | False | PMRC | Enable memory in output layer |

## 💡 Usage Patterns

### Pattern 1: Time Series Prediction

```python
# AMRC excels at sequences with consistent patterns
model = AMRC(input_size=10, hidden_size=50, output_size=1, beta=0.5, alpha=0.2)
model.fit(X_timeseries, y_timeseries, epochs=100)

# AMN for complex time series with long-range dependencies
model = AMN(input_size=10, hidden_size=64, output_size=1, 
            memory_manifold_size=128, memory_decay=0.998)
model.fit(X_timeseries, y_timeseries, epochs=100, batch_size=32)
```

### Pattern 2: Continuous Learning

```python
# Train on initial data
model = PMRC(input_size=20, hidden_size=40, output_size=10)
model.fit(X_initial, y_initial, epochs=50)

# Continue learning without losing previous knowledge
model.fit(X_new, y_new, epochs=20, reset_memory=False)
```

### Pattern 3: Transfer Learning

```python
# Phase 1: Train on general data
model = PMRC(input_size=100, hidden_size=200, output_size=50)
model.fit(X_general, y_general, epochs=100)

# Phase 2: Fine-tune for specific task
model.freeze_hidden_layer()  # Keep learned representations
model.fit(X_specific, y_specific, epochs=30)
model.unfreeze_hidden_layer()
```

### Pattern 4: Learning What to Remember (PMRC only)

```python
# Let the network discover important patterns
model = PMRC(
    input_size=50, 
    hidden_size=100, 
    output_size=20,
    use_learnable_gates=True  # Key feature!
)

model.fit(X_train, y_train, epochs=100)

# Inspect what it learned to remember
gates = model.get_gate_state()
print(f"Gate statistics: mean={gates.mean():.3f}, std={gates.std():.3f}")
```

### Pattern 5: Adaptive Memory for Complex Sequences (AMN only)

```python
# AMN automatically adapts memory mechanisms
model = AMN(
    input_size=50,
    hidden_size=128,
    output_size=20,
    memory_manifold_size=256,  # Large memory capacity
    dt=0.1,                    # Fine temporal control
    memory_decay=0.999         # Long-term memory retention
)

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Monitor adaptive behavior
print(f"Manifold Energy: {model.avg_manifold_energy:.4f}")
print(f"Avg LC Timescale: {model.avg_lc_timescale:.4f}")
print(f"Avg LRU Magnitude: {model.avg_lru_magnitude:.4f}")

# Make predictions with memory continuity
predictions = model.predict(X_test, reset_memory=False)
```

### Pattern 6: Partial Training by Magnitude (PMRC only)

```python
# Freeze small weights (likely less important)
model.freeze_by_magnitude(threshold=0.01, freeze_large=False)
model.fit(X_train, y_train, epochs=50)

# Or freeze large weights (preserve learned features)
model.freeze_by_magnitude(threshold=0.5, freeze_large=True)
model.fit(X_finetune, y_finetune, epochs=20)
```

## 🔧 Advanced Features

### Memory State Manipulation

```python
# Save current memory state
memory_snapshot = model.get_memory_state()

# Process some data
predictions = model.predict(X_test)

# Restore previous memory state
model.set_memory_state(memory_snapshot)

# Or start fresh
model.reset_memory()
```

### AMN-Specific: Manifold Memory Control

```python
# Get full manifold state
manifold = model.get_manifold_state()
print(f"Manifold shape: {manifold.shape}")

# Reset just the manifold (keep LRU/LC states)
model.reset_manifold()

# Reset everything
model.reset_memory()  # Resets both LRU and manifold
```

### Model Inspection (PMRC)

```python
# Check training progress
print(f"Training steps: {model.training_steps}")
print(f"Last loss: {model.last_loss:.6f}")
print(f"Avg memory magnitude: {model.avg_memory_magnitude:.6f}")

# Check feature usage
print(f"Using recurrent connections: {model.use_recurrent}")
print(f"Using learnable gates: {model.use_learnable_gates}")
print(f"Using output memory: {model.use_output_memory}")
```

### Model Inspection (AMN)

```python
# Check adaptive mechanisms
print(f"Training steps: {model.training_steps}")
print(f"Last loss: {model.last_loss:.6f}")
print(f"Manifold energy: {model.avg_manifold_energy:.6f}")
print(f"LRU magnitude: {model.avg_lru_magnitude:.6f}")
print(f"LC timescale: {model.avg_lc_timescale:.6f}")

# Detailed info
model.print_info()
```

### Dynamic Parameter Adjustment

```python
# AMRC/PMRC: Adjust memory parameters
model = AMRC(input_size=10, hidden_size=20, output_size=5, beta=0.1)
model.fit(X_initial, y_initial, epochs=50)

# Strengthen memory retention
model.beta = 0.7
model.fit(X_continuation, y_continuation, epochs=50)

# Adjust learning rate
model.learning_rate = 0.001

# AMN: Adjust temporal dynamics
model = AMN(input_size=10, hidden_size=32, output_size=5)
model.fit(X_initial, y_initial, epochs=50)

# Fine-tune temporal resolution
model.dt = 0.05  # Finer time steps
model.memory_decay = 0.9995  # Longer memory
model.learning_rate = 0.0005
model.fit(X_continuation, y_continuation, epochs=50)
```

## 🗂️ Architecture Comparison

| Feature | AMRC | PMRC | AMN |
|---------|------|------|-----|
| Memory preservation (β) | ✅ | ✅ | ❌* |
| Stateful neurons (α) | ✅ | ✅ | ❌* |
| Learnable memory gates | ❌ | ✅ | ✅ (implicit) |
| Output layer memory | ❌ | ✅ | ❌ |
| Basic layer freezing | ✅ | ✅ | ❌ |
| Advanced partial training | ❌ | ✅ | ❌ |
| Liquid Constant neurons | ❌ | ❌ | ✅ |
| Linear Recurrent Units | ❌ | ❌ | ✅ |
| Associative Memory Manifold | ❌ | ❌ | ✅ |
| Adaptive time constants | ❌ | ❌ | ✅ |
| Parallel-friendly | ❌ | ❌ | ✅ |
| Serialization formats | Binary | Binary/Joblib/Pickle | Binary |
| Performance | Fastest | Very Fast | Fast |
| Memory capacity | Good | Very Good | Excellent |
| Best for | Production, Speed | Research, Flexibility | Complex sequences, Long-range dependencies |

*AMN uses different memory mechanisms (LC, LRU, AMM) instead of explicit β/α parameters

## 📊 Performance Tips

1. **Start with AMRC** for simpler tasks and fastest training
2. **Use PMRC** when you need learnable memory dynamics or transfer learning
3. **Use AMN** for complex sequences with long-range dependencies or when you need adaptive time constants
4. **Set beta=0** (AMRC/PMRC) to disable memory and get a standard feedforward network
5. **Enable learnable gates** (PMRC) when patterns are complex and you're unsure what to remember
6. **Tune memory_manifold_size** (AMN): 32-64 for speed, 128-256 for capacity, 512+ for very long sequences
7. **Adjust dt** (AMN): smaller values (0.01-0.05) for fine temporal details, larger (0.1-0.5) for smoother dynamics
8. **Use partial training** (PMRC) to prevent catastrophic forgetting in continual learning scenarios
9. **Save models frequently** during long training runs
10. **Monitor AMN diagnostics** to understand what the network is learning

## ⚠️ Common Pitfalls

1. **Memory accumulation**: Reset memory between independent sequences
   ```python
   for sequence in sequences:
       model.reset_memory()
       predictions = model.predict(sequence)
   ```

2. **Type mismatch**: Always use `np.float32` for inputs
   ```python
   X = X.astype(np.float32)
   y = y.astype(np.float32)
   ```

3. **Frozen layers**: Remember to unfreeze when done with partial training (PMRC)
   ```python
   model.freeze_hidden_layer()
   # ... training ...
   model.unfreeze_hidden_layer()  # Don't forget!
   ```

4. **Size mismatch on load**: Ensure loaded model matches the size of saved model

5. **AMN batch size**: AMN requires a batch_size parameter in fit() - don't forget to specify it

6. **AMN memory decay**: Values too close to 1.0 (>0.9999) may cause numerical instability

## 🔍 Troubleshooting

**Problem**: `RuntimeError: AMRC implementation not available`
- **Solution**: Compile `memory_net_dll.c` first

**Problem**: `RuntimeError: PMRC implementation not available`
- **Solution**: Compile `memory_net_extended.c` first

**Problem**: `RuntimeError: AMN implementation not available`
- **Solution**: Compile `amn.c` with OpenMP support first

**Problem**: Memory doesn't seem to persist
- **Solution**: Check beta value (AMRC/PMRC: beta=0 means no memory), or check memory_decay (AMN: lower values = faster forgetting)

**Problem**: Training is slow
- **Solution**: Use AMRC instead of PMRC/AMN, or reduce hidden_size, or increase batch_size (AMN)

**Problem**: Model forgets previous knowledge
- **Solution**: Use partial training (PMRC: freeze layers) or don't reset memory, or increase memory_decay (AMN)

**Problem**: AMN gives inconsistent results
- **Solution**: Set a random_state for reproducibility, or reset_memory between independent predictions

**Problem**: AMN compilation fails on macOS
- **Solution**: Install OpenMP with `brew install libomp`, use the macOS-specific compilation command

## 📚 Further Reading

- **Theory**: See the C implementation headers for mathematical details
- **AMRC/PMRC**: Traditional memory-native architectures with explicit memory parameters
- **AMN**: Modern adaptive architecture combining Liquid Time Constants, Linear Recurrent Units, and Associative Memory
- **Examples**: Run `python sample.py` for comprehensive usage examples
- **Benchmarks**: Compare AMRC vs PMRC vs AMN on your specific task

## 🔬 Model Selection Guide

**Choose AMRC if you need:**
- Fast training and inference
- Simple, interpretable memory mechanism
- Production deployment with minimal overhead
- Basic transfer learning capabilities

**Choose PMRC if you need:**
- Fine control over what to remember (learnable gates)
- Advanced transfer learning (partial training by magnitude/percentage)
- Multiple serialization options
- Output layer memory

**Choose AMN if you need:**
- Adaptive time constants that respond to input importance
- Very long-range dependencies (100+ steps)
- Parallel-friendly recurrent processing
- Global memory for cross-sequence patterns
- Automatic adaptation to temporal dynamics
- Best performance on complex sequential tasks

## 📄 License

See project root for license information (GPL V3).

## 🤝 Contributing

Contributions welcome! Key areas:
- Additional memory architectures
- Performance optimizations
- Extended documentation
- Example notebooks
- Benchmarking suite

## 📞 Support

For issues, questions, or feature requests, please check:
1. This README
2. `sample.py` for working examples
3. Source code docstrings
4. C implementation comments
5. Run `python api.py` to check model availability

## 🆕 What's New in v1.0.0

- **New Model**: AMN (Adaptive Memory Network) with Liquid Constants, LRU, and Associative Memory
- **Unified API**: All three models (AMRC, PMRC, AMN) now share a common interface
- **Enhanced diagnostics**: Track manifold energy, LRU magnitude, and adaptive time constants (AMN)
- **Improved documentation**: Complete API reference with examples for all models
- **Factory function**: `create_model()` supports all three architectures

---

**Version**: 1.0.0  
**Last Updated**: February 2026  
**Models**: AMRC, PMRC, AMN