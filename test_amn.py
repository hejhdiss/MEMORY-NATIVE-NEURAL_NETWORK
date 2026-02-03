#Licensed under GPL V3.

import numpy as np
from api import AMN

# ============================================================
# 1. Reproducibility
# ============================================================
np.random.seed(42)

# ============================================================
# 2. Create SEQUENTIAL data (AMN-friendly)
# ============================================================
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

# ============================================================
# 3. Sequential train / test split (IMPORTANT)
# ============================================================
split = int(0.8 * n_samples)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

# ============================================================
# 4. Create AMN model
# ============================================================
model = AMN(
    input_size=input_size,
    hidden_size=32,
    output_size=output_size,
    memory_manifold_size=64,
    learning_rate=0.001,
    dt=0.1,
    memory_decay=0.998
)

# ============================================================
# 5. Train (memory flows normally)
# ============================================================
model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32,
    verbose=1
)

# ============================================================
# 6. Sequential evaluation (NO memory reset)
# ============================================================
y_pred = model.predict(
    X_test,
    reset_memory=False  # 🔑 KEY POINT
)

# ============================================================
# 7. Compute R² (sequential)
# ============================================================
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1.0 - ss_res / ss_tot

print(f"\nSequential R² score: {r2:.4f}")

# ============================================================
# 8. Diagnostics (AMN memory health)
# ============================================================
print("\nAMN Diagnostics:")
print(f"  Avg manifold energy : {model.avg_manifold_energy:.6f}")
print(f"  Avg LC timescale    : {model.avg_lc_timescale:.6f}")
