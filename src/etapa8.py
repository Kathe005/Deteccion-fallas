import os
import numpy as np
import matplotlib.pyplot as plt
from nucleo import *

np.random.seed(42)
patron = crear_patron()

def pieza_dificil(patron):
    """Defecto pequeno (2 a 5 px) y tenue (gris 150 a 190)."""
    lado = np.random.randint(2, 6)
    fila = np.random.randint(4, 64 - lado - 4)
    col = np.random.randint(4, 64 - lado - 4)
    valor = np.random.uniform(150, 190)
    return agregar_ruido(agregar_defecto(patron, fila, col, lado, valor))

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Datos: 300 buenas (0) y 300 malas (1)
X, y = [], []
for _ in range(300):
    X.append(caracteristica(agregar_ruido(patron), patron))
    y.append(0)
    X.append(caracteristica(pieza_dificil(patron), patron))
    y.append(1)
X = np.array(X)
y = np.array(y)

# Entrenamiento 70% / prueba 30%
orden = np.random.permutation(len(X))
corte = int(0.7 * len(X))
Xtr, ytr = X[orden[:corte]], y[orden[:corte]]
Xte, yte = X[orden[corte:]], y[orden[corte:]]

media, desv = Xtr.mean(), Xtr.std()
Xtr = (Xtr - media) / desv
Xte = (Xte - media) / desv

w, b, lr = 0.0, 0.0, 0.1
perdidas = []
for epoca in range(500):
    p = sigmoide(w * Xtr + b)
    perdidas.append(-np.mean(ytr * np.log(p + 1e-9) + (1 - ytr) * np.log(1 - p + 1e-9)))
    w -= lr * np.mean((p - ytr) * Xtr)
    b -= lr * np.mean(p - ytr)

pred = (sigmoide(w * Xte + b) >= 0.5).astype(int)

# Matriz de confusion hecha a mano
vn = np.sum((pred == 0) & (yte == 0))
fp = np.sum((pred == 1) & (yte == 0))
fn = np.sum((pred == 0) & (yte == 1))
vp = np.sum((pred == 1) & (yte == 1))
exactitud = (vn + vp) / len(yte)
precision = vp / (vp + fp) if (vp + fp) > 0 else 0
recall = vp / (vp + fn) if (vp + fn) > 0 else 0

print("Perdida inicial: %.4f   final: %.4f" % (perdidas[0], perdidas[-1]))
print("Matriz de confusion (filas = real, columnas = prediccion):")
print("              Aprobada  Rechazada")
print("Real buena    %6d    %6d" % (vn, fp))
print("Real mala     %6d    %6d" % (fn, vp))
print("Exactitud: %.3f   Precision: %.3f   Recall: %.3f" % (exactitud, precision, recall))
print("")
print("Comparacion de umbrales de decision:")
prob = sigmoide(w * Xte + b)
for u in [0.5, 0.3, 0.1]:
    pr = (prob >= u).astype(int)
    fp_u = np.sum((pr == 1) & (yte == 0))
    fn_u = np.sum((pr == 0) & (yte == 1))
    print("umbral %.1f -> buenas rechazadas: %d   malas aprobadas: %d" % (u, fp_u, fn_u))

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
ejes[0].plot(perdidas)
ejes[0].set_title("Perdida vs epoca (problema dificil)")
ejes[0].set_xlabel("Epoca")
ejes[0].set_ylabel("Perdida")
matriz = np.array([[vn, fp], [fn, vp]])
ejes[1].imshow(matriz, cmap="Blues")
for i in range(2):
    for j in range(2):
        ejes[1].text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=16)
ejes[1].set_xticks([0, 1])
ejes[1].set_xticklabels(["Aprobada", "Rechazada"])
ejes[1].set_yticks([0, 1])
ejes[1].set_yticklabels(["Real buena", "Real mala"])
ejes[1].set_title("Matriz de confusion")

os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa8_dificil.png", dpi=150)
plt.show()