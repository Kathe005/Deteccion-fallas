import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from nucleo import *

np.random.seed(7)
patron = crear_patron()

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ---------- Entrenamiento (igual que la etapa 5) ----------
X, y = [], []
for _ in range(150):
    X.append(caracteristica(agregar_ruido(patron), patron))
    y.append(0)
    X.append(caracteristica(pieza_mala(patron), patron))
    y.append(1)
X = np.array(X)
y = np.array(y)

media, desv = X.mean(), X.std()
Xn = (X - media) / desv

w, b, lr = 0.0, 0.0, 0.1
for epoca in range(300):
    p = sigmoide(w * Xn + b)
    w -= lr * np.mean((p - y) * Xn)
    b -= lr * np.mean(p - y)

def probabilidad_mala(img):
    x = (caracteristica(img, patron) - media) / desv
    return sigmoide(w * x + b)

# ---------- Cinta transportadora ----------
N = 5
conteo = {"aprobadas": 0, "rechazadas": 0}
cola = []

def nueva_pieza():
    if np.random.rand() < 0.35:
        img = pieza_mala(patron)
    else:
        img = agregar_ruido(patron)
    p = probabilidad_mala(img)
    if p >= 0.5:
        conteo["rechazadas"] += 1
    else:
        conteo["aprobadas"] += 1
    return img, p

fig, ejes = plt.subplots(1, N, figsize=(13, 4))

def actualizar(frame):
    cola.append(nueva_pieza())
    if len(cola) > N:
        cola.pop(0)
    for ax in ejes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    for k, (img, p) in enumerate(cola):
        ax = ejes[k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        rechazada = p >= 0.5
        color = "red" if rechazada else "green"
        texto = "RECHAZADO" if rechazada else "APROBADO"
        ax.set_title(texto + "\nP(mala) = %.2f" % p, color=color)
        for borde in ax.spines.values():
            borde.set_edgecolor(color)
            borde.set_linewidth(5)
    fig.suptitle("Aprobadas: %d    Rechazadas: %d"
                 % (conteo["aprobadas"], conteo["rechazadas"]))

os.makedirs("resultados", exist_ok=True)

anim = FuncAnimation(
    fig, 
    actualizar, 
    frames=40, 
    interval=1000, 
    repeat=False
    )

plt.show()

fig.savefig("resultados/etapa6_cinta.png", dpi=150)