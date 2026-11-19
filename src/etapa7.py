import os
import numpy as np
import matplotlib.pyplot as plt
import os
from nucleo import *

np.random.seed(42)
patron = crear_patron()

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Datos: 150 buenas (y=0) y 150 malas (y=1)
X, y = [], []
for _ in range(150):
    X.append(caracteristica(agregar_ruido(patron), patron))
    y.append(0)
    X.append(caracteristica(pieza_mala(patron), patron))
    y.append(1)
X = np.array(X)
y = np.array(y)

# Normalizar
Xn = (X - X.mean()) / X.std()

def entrenar(Xn, y, lr, epocas=300):
    w, b = 0.0, 0.0
    perdidas = []
    for _ in range(epocas):
        p = sigmoide(w * Xn + b)
        perdida = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        perdidas.append(perdida)
        w -= lr * np.mean((p - y) * Xn)
        b -= lr * np.mean(p - y)
    return w, b, perdidas

plt.figure(figsize=(8, 5))
for lr in [0.01, 0.1, 1.0]:
    w, b, perdidas = entrenar(Xn, y, lr)
    print("lr = %-5s  perdida final = %.4f   w = %.3f   b = %.3f"
          % (lr, perdidas[-1], w, b))
    plt.plot(perdidas, label="lr = %s" % lr)

plt.xlabel("Epoca")
plt.ylabel("Perdida")
plt.title("Convergencia segun la tasa de aprendizaje")
plt.legend()
plt.grid(alpha=0.3)

os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/convergencia_lr.png", dpi=150)
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa2_diferencia.png", dpi=150)
plt.show()