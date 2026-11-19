import numpy as np
import matplotlib.pyplot as plt
import os
def crear_patron(tam=64):
    """Azulejo ideal: fondo claro, borde oscuro y un cuadro central."""
    img = np.full((tam, tam), 200.0)
    img[0:3, :] = 80
    img[-3:, :] = 80
    img[:, 0:3] = 80
    img[:, -3:] = 80
    img[24:40, 24:40] = 120
    return img

def agregar_ruido(img, sigma=3):
    """Pequeña variación normal, como en una cámara real."""
    return img + np.random.normal(0, sigma, img.shape)

def agregar_defecto(img, fila, col, lado=6, valor=30):
    """Mancha oscura cuadrada en la posición indicada."""
    salida = img.copy()
    salida[fila:fila + lado, col:col + lado] = valor
    return salida

patron = crear_patron()
buena = agregar_ruido(patron)
mala = agregar_ruido(agregar_defecto(patron, 10, 45))

# Resta matricial y MSE
dif_buena = buena - patron
dif_mala = mala - patron
mse_buena = np.mean(dif_buena ** 2)
mse_mala = np.mean(dif_mala ** 2)

print("MSE pieza buena:", mse_buena)
print("MSE pieza mala: ", mse_mala)

# Visualización
fig, ejes = plt.subplots(1, 4, figsize=(14, 4))
ejes[0].imshow(patron, cmap="gray", vmin=0, vmax=255)
ejes[0].set_title("Patron ideal")
ejes[1].imshow(buena, cmap="gray", vmin=0, vmax=255)
ejes[1].set_title("Pieza buena")
ejes[2].imshow(mala, cmap="gray", vmin=0, vmax=255)
ejes[2].set_title("Pieza mala")
ejes[3].imshow(np.abs(dif_mala), cmap="hot")
ejes[3].set_title("Diferencia (mala)")
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa2_diferencia.png", dpi=150)
plt.show()