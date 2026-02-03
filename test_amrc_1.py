#Licensed under GPL V3.

import numpy as np
from api import AMRC

np.random.seed(0)

n = 500
input_size = 3

X = np.random.randn(n, input_size).astype(np.float32)

y = (
    0.8 * X[:, 0]
    - 0.5 * X[:, 1]
).reshape(-1, 1).astype(np.float32)

split = int(0.8 * n)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = AMRC(
    input_size=input_size,
    hidden_size=16,
    output_size=1,
    beta=0.1,
    alpha=0.1,
    learning_rate=0.01
)

model.fit(X_train, y_train, epochs=40, verbose=1)
y_pred = model.predict(X_test)

ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

print("AMRC R² (baseline):", 1 - ss_res / ss_tot)
