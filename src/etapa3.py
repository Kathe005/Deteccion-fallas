import numpy as np
import matplotlib.pyplot as plt
import os

def crear_patron(tam=64):
    img = np.full((tam, tam), 200.0)
    img[0:3, :] = 80
    img[-3:, :] = 80
    img[:, 0:3] = 80
    img[:, -3:] = 80
    img[24:40, 24:40] = 120
    return img

def agregar_ruido(img, sigma=3):
    return img + np.random.normal(0, sigma, img.shape)

def agregar_defecto(img, fila, col, lado=6, valor=30):
    salida = img.copy()
    salida[fila:fila + lado, col:col + lado] = valor
    return salida

def mse_por_bloques(prueba, patron, bloque=8):
    """Devuelve una matriz con el MSE de cada bloque."""
    filas = prueba.shape[0] // bloque
    cols = prueba.shape[1] // bloque
    mapa = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            f0, f1 = i * bloque, (i + 1) * bloque
            c0, c1 = j * bloque, (j + 1) * bloque
            diferencia = prueba[f0:f1, c0:c1] - patron[f0:f1, c0:c1]
            mapa[i, j] = np.mean(diferencia ** 2)
    return mapa

patron = crear_patron()
mala = agregar_ruido(agregar_defecto(patron, 10, 45))
mapa = mse_por_bloques(mala, patron)

print("Forma del mapa:", mapa.shape)
print("MSE maximo:", mapa.max())
print("Bloque con el maximo (fila, col):", np.unravel_index(mapa.argmax(), mapa.shape))

fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
ejes[0].imshow(mala, cmap="gray", vmin=0, vmax=255)
ejes[0].set_title("Pieza mala")
im = ejes[1].imshow(mapa, cmap="hot")
ejes[1].set_title("MSE por bloque")
plt.colorbar(im, ax=ejes[1])
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa3_mapa_bloques.png", dpi=150)
plt.show()