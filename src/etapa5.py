import numpy as np
import matplotlib.pyplot as plt
import os
from nucleo import *

np.random.seed(42)
patron = crear_patron()

# Datos: 150 buenas (y=0) y 150 malas (y=1)
X = []
y = []
for _ in range(150):
    X.append(caracteristica(agregar_ruido(patron), patron))
    y.append(0)
    X.append(caracteristica(pieza_mala(patron), patron))
    y.append(1)
X = np.array(X)
y = np.array(y)

# Division entrenamiento (70%) / prueba (30%)
orden = np.random.permutation(len(X))
corte = int(0.7 * len(X))
Xtr, ytr = X[orden[:corte]], y[orden[:corte]]
Xte, yte = X[orden[corte:]], y[orden[corte:]]

# Normalizar con las estadisticas del entrenamiento
media, desv = Xtr.mean(), Xtr.std()
Xtr = (Xtr - media) / desv
Xte = (Xte - media) / desv

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Entrenamiento: una neurona, descenso de gradiente
w, b = 0.0, 0.0
lr = 0.01
perdidas = []
for epoca in range(300):
    p = sigmoide(w * Xtr + b)
    perdida = -np.mean(ytr * np.log(p + 1e-9) + (1 - ytr) * np.log(1 - p + 1e-9))
    perdidas.append(perdida)
    dw = np.mean((p - ytr) * Xtr)
    db = np.mean(p - ytr)
    w -= lr * dw
    b -= lr * db

pred = sigmoide(w * Xte + b) >= 0.5
print("w =", round(w, 3), " b =", round(b, 3))
print("Perdida inicial:", round(perdidas[0], 4))
print("Perdida final:  ", round(perdidas[-1], 4))
print("Exactitud en prueba:", np.mean(pred == yte))

plt.plot(perdidas)
plt.xlabel("Epoca")
plt.ylabel("Perdida")
plt.title("Convergencia del perceptron")
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa2_diferencia.png", dpi=150)
plt.show()