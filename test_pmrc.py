#Licensed under GPL V3.

import numpy as np
from api import PMRC

# =====================================================
# 1. Same sequential data
# =====================================================
np.random.seed(42)

n_samples = 600
input_size = 5
output_size = 1

t = np.linspace(0, 12, n_samples)

X = np.zeros((n_samples, input_size), dtype=np.float32)
X[:, 0] = np.sin(t)
X[:, 1] = np.cos(0.5 * t)
X[:, 2] = np.tanh(t / 3)
X[:, 3] = np.random.randn(n_samples) * 0.05
X[:, 4] = np.roll(X[:, 0], 1)

y = (
    0.7 * X[:, 0]
    - 0.4 * X[:, 1]
    + 0.3 * np.sin(0.2 * t)
).reshape(-1, 1).astype(np.float32)

# =====================================================
# 2. Sequential split
# =====================================================
split = int(0.8 * n_samples)

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# =====================================================
# 3. PMRC model (api.py accurate)
# =====================================================
model = PMRC(
    input_size=input_size,
    hidden_size=32,
    output_size=output_size,
    beta=0.4,
    alpha=0.1,
    learning_rate=0.01,
    use_learnable_gates=True
)

# =====================================================
# 4. Train
# =====================================================
model.fit(
    X_train,
    y_train,
    epochs=60,
    verbose=1
)

# =====================================================
# 5. Sequential prediction (NO reset)
# =====================================================
y_pred = model.predict(X_test)

# =====================================================
# 6. Sequential R²
# =====================================================
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1.0 - ss_res / ss_tot

print(f"\nPMRC Sequential R²: {r2:.4f}")

# Optional PMRC diagnostics
print(f"Avg gate value: {model.avg_gate_value:.4f}")
print(f"Avg memory magnitude: {model.avg_memory_magnitude:.4f}")
