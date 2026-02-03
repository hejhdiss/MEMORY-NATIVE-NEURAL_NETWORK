# 🧠 MEMORY-NATIVE NEURAL NETWORK

**Beyond External Storage: What if AI Could Remember Like We Do?**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Made with Claude](https://img.shields.io/badge/Made%20with-Claude%20Sonnet%204.5-blueviolet.svg)](https://claude.ai)

> **A crazy experiment in neural architecture:** What if memory wasn't bolted on as an afterthought, but was native to the very neurons themselves?

This repository contains **three novel memory-native neural network architectures** that challenge conventional thinking about how AI models should handle memory and temporal information.

📖 **Read the full story**: [Beyond External Storage: What if AI Could Remember Like We Do?](https://dev.to/hejhdiss/beyond-external-storage-what-if-ai-could-remember-like-we-do-458j)

---

## 🎯 What Makes This Different?

Traditional neural networks treat memory as an external component—something you add via LSTMs, attention mechanisms, or vector databases. But what if memory was **intrinsic** to the neurons themselves?

This project implements three increasingly sophisticated approaches:

1. **AMRC** (Adaptive Memory Recurrent Cell) - Basic memory-native neurons with learnable retention
2. **PMRC** (Persistent Memory Recurrent Cell) - Adds learnable gates and advanced partial training
3. **AMN** (Adaptive Memory Network) - Combines Liquid Time Constants, Linear Recurrent Units, and Associative Memory Manifolds

All three models share a philosophy: **memory is not a feature, it's fundamental**.

---

## ⚡ Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK.git
cd MEMORY-NATIVE-NEURAL_NETWORK
```

2. **Compile the C libraries** (choose based on your OS):

```bash
# AMRC (required)
gcc -shared -o memory_net.dll memory_net_dll.c -lm -O3        # Windows
gcc -shared -fPIC -o memory_net.so memory_net_dll.c -lm -O3   # Linux/macOS

# PMRC (required)
gcc -shared -o memory_net_extended.dll memory_net_extended.c -lm -O3        # Windows
gcc -shared -fPIC -o memory_net_extended.so memory_net_extended.c -lm -O3   # Linux/macOS

# AMN (requires OpenMP)
gcc -shared -o amn.dll amn.c -lm -O3 -fopenmp        # Windows
gcc -shared -fPIC -o amn.so amn.c -lm -O3 -fopenmp   # Linux
gcc -shared -fPIC -o amn.dylib amn.c -lm -O3 -Xpreprocessor -fopenmp -lomp # macOS
```

3. **Install Python dependencies:**
```bash
pip install numpy joblib
```

### Your First Memory-Native Model

```python
import numpy as np
from api import AMRC  # Recommended: Always use api.py for a clean interface

# Create a memory-native model
model = AMRC(
    input_size=10,
    hidden_size=32,
    output_size=5,
    beta=0.3,        # Memory strength
    alpha=0.1        # Update rate
)

# Generate some data
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randn(100, 5).astype(np.float32)

# Train
model.fit(X_train, y_train, epochs=50, verbose=1)

# Predict
X_test = np.random.randn(20, 10).astype(np.float32)
predictions = model.predict(X_test)

# Save your trained model
model.save('my_memory_model.bin')
```

**That's it!** You've just trained a neural network where every neuron has its own internal memory.

---
## 🧬 Independent "Cousin" Architectures

The following models are experimental **"cousins"** of the standard AMN.  
These are **not included** in the main `api.py` or the unified API wrapper.

They are **specialized architectures** that must be used as **standalone `.py` scripts** and require their **own C libraries** to be compiled separately.

---

## 1. DTPN (Dual-Track Persistence Network)

### Architecture — *The "Universal Persistence" Hybrid*

DTPN is the most comprehensive memory model in the collection.  
It bridges the gap between the standard AMN and Extended Memory-Native concepts by providing **three distinct layers of data retention**:

- **Track 1: The Echo**  
  *Memory-Preserving Activation*  
  Retains a fraction of the immediate previous output (`β` factor) for temporal fluidity.

- **Track 2: The State**  
  *Stateful Neurons*  
  Individual neurons maintain a decaying internal **reservoir** of information (`α` factor).

- **Track 3: The Manifold**  
  *Global Memory Matrix*  
  A shared **associative whiteboard** for long-term contextual memory.

### Python API (`dtpn.py`)
- Provides a `DualTrackPersistenceNetwork` class
- Interfaces with `dtpn.dll` / `dtpn.so`
- **Configurable persistence** across all three tracks (Echo, State, Matrix)

### Use Case
Ideal for complex temporal tasks where the network must remember:
- short-term fluctuations (“jitters”)
- medium-term state
- long-term factual context simultaneously

---

## 2. Hyper-AMN (Multi-Head Associative Manifold)

### Architecture
Features a specialized **multi-head system** where different *manifolds* track specific data domains:

- **Spatial Manifold** – Positional and structural patterns  
- **Emotional Manifold** – Sentiment and tone patterns  
- **Logical Manifold** – Reasoning and causal patterns  

### Python API (`hyper-amn.py`)
- Includes a **Head Gating Mechanism** to route information to the most relevant manifold
- Supports `get_head_activations()` to monitor which specialized manifold is currently dominant

---

## 3. SGW-AMN (Sparse Global Workspace)

### Architecture
Inspired by **Global Workspace Theory** of consciousness.

- Uses a tiny **"conscious" bottleneck** where thousands of neurons compete
- **Competitive Routing** – Only the most vital information survives compression
- **Attention by Compression** – Forces extraction of essential features

### Python API (`sgw-amn.py`)
- Provides:
  - `get_workspace_sparsity()`
  - `get_competition_entropy()` to measure routing competition
- Includes an **information bottleneck rate** statistic showing the compression ratio

---

## 4. NDM (Neural Differential Manifolds)

### Architecture
Implements **Continuous Weight Evolution** using **Ordinary Differential Equations (ODEs)**.

- **True Neuroplasticity** – Weights evolve in real time (`dW/dt`) based on data importance
- **Hebbian Traces** – “Neurons that fire together wire together” guide weight evolution

### Python API (`ndm.py`)
- Includes `get_avg_weight_velocity()` to track rewiring speed
- Includes `get_avg_plasticity()` to measure how fluid or rigid the learning state is

---

## Summary of Differences

| Architecture | Key Innovation | Best For |
|-------------|---------------|----------|
| **DTPN** | Echo + State + Manifold | Maximum data persistence across all time scales |
| **Hyper-AMN** | Multi-Head Manifolds | Categorical separation (Logic vs Emotion vs Space) |
| **SGW-AMN** | Competitive Bottleneck | Feature extraction and “conscious” focus |
| **NDM** | ODE-based Weight Evolution | Environments with constantly changing rules |

---

## 🚀 Try It Out!

### Recommended: Use the High-Level API

**We strongly recommend using `api.py`** for all interactions with these models. It provides:
- Clean, unified interface across all three architectures
- Automatic error handling and validation
- Consistent save/load functionality
- Better documentation and type hints

```python
from api import AMRC, PMRC, AMN, create_model

# All models share the same interface
model = create_model('amn', input_size=10, hidden_size=32, output_size=5)
```

### Run the Examples

Check out `sample.py` for comprehensive examples covering:
- Basic usage of all three models (AMRC, PMRC, **AMN**)
- Memory state manipulation
- Partial training and transfer learning
- Model persistence
- Advanced features like learnable gates, manifold memory, and adaptive time constants

```bash
python sample.py
```

### Run the Tests

Validate the implementation with comprehensive tests:

```bash
python test_per.py
```

This runs bit-perfect validation of:
- State persistence across save/load
- Partial training compatibility
- Memory state restoration
- Error handling

---

## 🏗️ Architecture Overview

### AMRC - The Foundation
```
Input → [Memory-Native Neurons] → Output
         ↑
         Memory (β, α parameters)
```
- **β (beta)**: How much of the past to retain
- **α (alpha)**: How fast memory updates
- **Features**: Partial training, layer freezing

### PMRC - The Extension
```
Input → [Learnable Gates] → [Memory Neurons] → [Output Memory] → Output
         ↑                    ↑                  ↑
         What to remember     How to remember    Output-level memory
```
- Everything in AMRC, plus:
- **Learnable Gates**: Network decides what's important
- **Output Memory**: Memory at every layer
- **Advanced Partial Training**: Freeze by percentage, magnitude, or specific neurons

### AMN - The Fusion ⭐ NEW
```
Input → [Liquid Constant] → [Linear Recurrent Unit] → Output
         ↓                    ↓                         ↑
         Adaptive τ           Parallel-friendly         │
         │                    │                         │
         └────────────────────┴─→ [Associative Memory Manifold]
                                  Global Knowledge Base
```
- **Liquid Constants (LC)**: Neurons with adaptive time scales
- **Linear Recurrent Units (LRU)**: Efficient recurrent processing
- **Associative Memory Manifolds (AMM)**: Global memory whiteboard
- **Automatic Adaptation**: Time constants adjust based on input importance

---

## 📚 Documentation

- **[USAGE.md](USAGE.md)**: Complete API reference and detailed examples
- **[sample.py](sample.py)**: Working examples for all three models
- **[test_per.py](test_per.py)**: Comprehensive test suite
- **C implementations**: See `.c` files for mathematical details

---
## 🧪 Specialized Testing Suite

Because these models utilize **memory-native architectures** and **custom C-compiled backends**, standard unit tests aren't enough.  
This repository includes **specialized testing scripts** designed to validate the **unique temporal and persistent nature** of these networks.

---

## 🔍 Test Files Overview

| Test File         | Focus Area            | What it Validates |
|-------------------|----------------------|-------------------|
| `test_per.py`     | State Persistence     | Ensures models maintain **bit-perfect accuracy** after saving and loading `.bin` files |
| `test_amrc_1.py`  | Baseline Regression   | Tests the basic **AMRC cell**’s ability to solve standard linear and non-linear mappings |
| `test_amrc_2.py`  | Temporal Carry        | Validates the **memory carry effect** — checks whether past states influence future predictions |
| `test_pmrc.py`    | Gated Memory          | Tests **learnable memory gates** and diagnostic outputs (average gate value, memory magnitude) |
| `test_amn.py`     | Manifold Dynamics     | Focuses on **AMN Liquid Time Constants** and **Associative Memory Manifold health** |

---

## 🛠️ Key Testing Features

### 1. Bit-Perfect Persistence

Unlike standard neural networks, **memory-native models store internal state**, including:

- Memory preservation factor (`β`)
- Memory update rate (`α`)
- Hidden memory manifolds

These tests verify that:

- Memory states are captured **exactly** during `save()`
- Reloaded models produce an **identical R² score** as the original model

---

### 2. Sequential Validation (No-Reset Testing)

For **AMN** and **PMRC**, correct memory flow across time is critical.

- **`reset_memory=False` flag**  
  Ensures the internal hidden state is carried from training into prediction.

- **Chronological Data Splitting**  
  Data is split sequentially (not randomly) so the model cannot *peek into the future*.

---

### 3. Diagnostic Monitoring

These tests go beyond simple **pass/fail** checks and actively monitor **memory health**:

- **Manifold Energy**  
  Detects whether AMN global memory is saturating or collapsing.

- **Liquid Constant (LC) Timescale**  
  Monitors how Liquid Time Constants adapt to input frequency.

- **Gate Values (PMRC)**  
  Verifies whether memory gates are genuinely learning to open/close rather than remaining static.

---

## 🚀 How to Run Tests

To run the full validation suite and verify your **local C-compiled backend**, execute:

```bash
# Run the primary persistence and error-handling test
python test_per.py

# Run specific architecture benchmarks
python test_amrc_2.py
python test_pmrc.py
python test_amn.py
```

---

## 🎨 Example Use Cases

### 1. Time Series Prediction
```python
from api import AMN

model = AMN(
    input_size=1,
    hidden_size=64,
    output_size=1,
    memory_manifold_size=128  # Large memory for patterns
)
model.fit(time_series_data, targets, epochs=100, batch_size=32)
```

### 2. Transfer Learning with PMRC
```python
from api import PMRC

model = PMRC(input_size=10, hidden_size=32, output_size=5)
model.fit(general_data, general_targets, epochs=50)

# Freeze learned features, train only output
model.freeze_hidden_layer()
model.fit(specific_data, specific_targets, epochs=20)
```

### 3. Continuous Learning
```python
from api import AMRC

model = AMRC(input_size=8, hidden_size=16, output_size=3)

# Learn from stream of data without catastrophic forgetting
for batch in data_stream:
    model.fit(batch.X, batch.y, epochs=1, reset_memory=False)
```

---

## 🔬 Why This Is Crazy (In a Good Way)

**Conventional wisdom** says:
- Memory should be in the architecture (LSTM gates, attention heads)
- You need external storage for long-term memory
- Neurons should be stateless

**This project argues**:
- What if memory was in the neurons themselves?
- What if each neuron could decide what to remember?
- What if time constants could adapt based on what's important?

The results? Models that:
- ✅ Learn what to remember without being told
- ✅ Adapt their temporal dynamics automatically
- ✅ Maintain both short and long-range dependencies
- ✅ Can be partially trained without catastrophic forgetting

---

## 🤖 Generated with Claude Sonnet 4.5

**Yes, all of this code was generated with Claude Sonnet 4.5!**

This entire project—from the mathematical formulations in C to the Python APIs to the comprehensive test suite—was created through conversations with Claude. It demonstrates:

- How AI can help explore novel architectural ideas
- The power of iterative refinement through conversation
- That crazy ideas can become working code

Want to experiment like this yourself? Check out [Claude.ai](https://claude.ai) and start building!

---

## 🎯 When to Use Each Model

| Model | Best For | Speed | Memory Capacity | Complexity |
|-------|----------|-------|-----------------|------------|
| **AMRC** | Production, Real-time | ⚡⚡⚡ | Good | Low |
| **PMRC** | Research, Transfer Learning | ⚡⚡ | Very Good | Medium |
| **AMN** | Complex Sequences, Long-range | ⚡ | Excellent | High |

**General Rule**: Start with AMRC for simplicity, move to PMRC for flexibility, use AMN for complex temporal patterns.

---

## 📊 Model Comparison

| Feature | AMRC | PMRC | AMN |
|---------|------|------|-----|
| Explicit memory (β, α) | ✅ | ✅ | ❌ |
| Learnable gates | ❌ | ✅ | ✅ (implicit) |
| Adaptive time constants | ❌ | ❌ | ✅ |
| Global memory manifold | ❌ | ❌ | ✅ |
| Partial training | Basic | Advanced | ❌ |
| Parallel-friendly | ❌ | ❌ | ✅ |

---

## 🛠️ Advanced Features

### Memory State Control
```python
# Save memory snapshot
snapshot = model.get_memory_state()

# Process data
predictions = model.predict(data)

# Restore or reset
model.set_memory_state(snapshot)  # Restore
model.reset_memory()               # Clear
```

### AMN-Specific: Manifold Inspection
```python
# Check what the network is learning
print(f"Manifold Energy: {model.avg_manifold_energy}")
print(f"LC Timescale: {model.avg_lc_timescale}")
print(f"LRU Magnitude: {model.avg_lru_magnitude}")

# Get full manifold state
manifold = model.get_manifold_state()
```

### Partial Training (PMRC)
```python
# Freeze by percentage
model.freeze_hidden_percentage(0.5)  # Freeze 50% of neurons

# Freeze by magnitude
model.freeze_by_magnitude(threshold=0.1)

# Freeze specific components
model.freeze_memory_gates()
model.freeze_hidden_layer()
```

---

## 🐛 Common Pitfalls

1. **Always use `np.float32`** for inputs and targets
2. **Reset memory** between independent sequences
3. **Remember to unfreeze** layers after partial training (PMRC)
4. **Specify batch_size** when using AMN
5. **Use the api.py interface** instead of calling C libraries directly

---

## 📈 Performance Tips

- Start with **small hidden sizes** (16-32) and scale up
- Use **AMRC** for speed-critical applications
- Use **AMN** for complex temporal patterns with long dependencies
- **Tune memory_manifold_size** (AMN): 32-64 for speed, 256+ for capacity
- **Monitor diagnostics** to understand what's being learned
- **Save frequently** during long training runs

---

## 🔬 Research & Experimentation

This is an **experimental project**—a testbed for ideas about memory-native architectures. The goal isn't to replace transformers or LSTMs, but to explore:

- Alternative ways to handle memory in neural networks
- Whether intrinsic neuron-level memory can compete with architectural memory
- How adaptive time constants affect learning dynamics
- The interplay between local (neuron) and global (manifold) memory

**We encourage you to**:
- Experiment with different architectures
- Benchmark against your use cases
- Propose improvements and variations
- Share your results

---

## 👨‍💻 Author & Experiment

**Author**: [@hejhdiss](https://github.com/hejhdiss) (Muhammed Shafin P)

This was a **crazy experiment**—and I mean that in the best way possible! 

While the code was generated with Claude Sonnet 4.5, this is **my experiment** exploring different varieties of memory-native architectures. I manually edited some parts, tested everything myself, and went through countless iterations to make this work.

**This is experimental research-grade code.** It's meant to be played with, broken, and rebuilt. If you're into neural network research or just love experimenting with novel architectures, this is for you!

### ⚗️ Why This Experiment Matters

- It challenges conventional thinking about memory in neural networks
- It explores whether memory should be architectural or intrinsic
- It combines multiple cutting-edge mechanisms (Liquid Constants, LRU, Associative Memory)
- It's a testbed for ideas that might inform future research

**For researchers**: I think this could be helpful for exploring memory-native approaches, continual learning, and adaptive temporal dynamics. The code is here to be studied, modified, and improved.

**For experimenters**: This is a playground. Break it. Improve it. Try crazy things. That's what it's for!

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Performance optimizations
- Additional memory mechanisms
- Extended benchmarks and comparisons
- Documentation improvements
- Bug fixes and edge cases

Feel free to fork, experiment, and submit PRs. This is an open experiment, and all ideas are welcome!

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE](LICENSE) file for full details.

In brief:
- ✅ Use, modify, and distribute freely
- ✅ Use in commercial products
- ⚠️ Must disclose source and use same license
- ⚠️ Changes must be documented

---

## 🎓 Learn More

- 📖 **Deep Dive**: [Beyond External Storage: What if AI Could Remember Like We Do?](https://dev.to/hejhdiss/beyond-external-storage-what-if-ai-could-remember-like-we-do-458j)
- 📚 **API Docs**: See [USAGE.md](USAGE.md) for complete reference
- 💻 **Examples**: Run `python sample.py` to see all features in action
- 🧪 **Tests**: Run `python test_per.py` for validation
- 🤖 **Created with**: [Claude Sonnet 4.5](https://claude.ai)

---

## ⭐ Star This Project

If you find this project interesting, please give it a star! It helps others discover these experimental architectures.

---

## 📬 Questions?

- Check [USAGE.md](USAGE.md) for detailed documentation
- Review [sample.py](sample.py) for working examples
- Run [test_per.py](test_per.py) to validate your setup
- Open an issue for bugs or feature requests

---

**Built with curiosity, powered by Claude Sonnet 4.5, driven by the question:**
> *What if memory wasn't something we added to neural networks, but something that was already there, waiting to be discovered?*