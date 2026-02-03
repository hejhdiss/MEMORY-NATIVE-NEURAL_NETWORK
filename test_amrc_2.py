import numpy as np
from api import AMRC

np.random.seed(2)

n = 500
X = np.random.randn(n, 2).astype(np.float32)

y = np.zeros((n, 1), dtype=np.float32)
for i in range(1, n):
    y[i] = 0.7 * y[i - 1] + 0.3 * X[i, 0]

split = int(0.8 * n)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = AMRC(
    input_size=2,
    hidden_size=20,
    output_size=1,
    beta=0.5,
    alpha=0.1,
    learning_rate=0.01
)

model.fit(X_train, y_train, epochs=60, verbose=1)
y_pred = model.predict(X_test)

ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

print("AMRC R² (memory carry):", 1 - ss_res / ss_tot)
