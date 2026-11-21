import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from nucleo import mse_por_bloques

TAM = 128      # todas las fotos se reducen a 128 x 128 pixeles
BLOQUE = 16    # 128 / 16 = cuadricula de 8 x 8 bloques

def suavizar(a):
    """Filtro de media 3x3 hecho a mano (convolucion con slicing)."""
    p = np.pad(a, 1, mode="edge")
    s = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            s += p[i:i + a.shape[0], j:j + a.shape[1]]
    return s / 9

def cargar_imagen(ruta):
    # Pillow solo abre y reduce la foto; el resto es NumPy
    img = Image.open(ruta).convert("L").resize((TAM, TAM))
    a = np.array(img, dtype=float)
    a = suavizar(a)
    # normalizar brillo y contraste para reducir el efecto de la luz
    return (a - a.mean()) / a.std()

patron = cargar_imagen("imagenes/patron.jpg")

resultados = []
for nombre in sorted(os.listdir("imagenes")):
    if nombre == "patron.jpg":
        continue
    if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cargar_imagen(os.path.join("imagenes", nombre))
    mapa = mse_por_bloques(img, patron, BLOQUE)
    resultados.append((nombre, mapa.max(), mapa.mean(), img, mapa))

print("%-12s %18s %12s" % ("foto", "MSE max por bloque", "MSE medio"))
for nombre, mx, md, img, mapa in resultados:
    print("%-12s %18.3f %12.3f" % (nombre, mx, md))

# Figura: fila 1 = fotos, fila 2 = mapa de error por bloques
# Umbral: punto medio entre la peor "buena" y la mejor "mala"
buenas = [mx for nombre, mx, md, img, mapa in resultados if nombre.startswith("buena")]
malas = [mx for nombre, mx, md, img, mapa in resultados if nombre.startswith("mala")]
print("")
if max(buenas) >= min(malas):
    print("Aviso: buenas y malas se solapan, no hay umbral que las separe.")
    umbral = None
else:
    umbral = (max(buenas) + min(malas)) / 2
    print("Umbral = %.2f (punto medio entre %.2f y %.2f)" % (umbral, max(buenas), min(malas)))
    for nombre, mx, md, img, mapa in resultados:
        veredicto = "RECHAZADO" if mx >= umbral else "APROBADO"
        print("%-12s %s" % (nombre, veredicto))
        
vmax = max(r[1] for r in resultados)
n = len(resultados)
fig, ejes = plt.subplots(2, n + 1, figsize=(3 * (n + 1), 6))
ejes[0, 0].imshow(patron, cmap="gray")
ejes[0, 0].set_title("patron")
ejes[1, 0].axis("off")
for k, (nombre, mx, md, img, mapa) in enumerate(resultados, start=1):
    ejes[0, k].imshow(img, cmap="gray")
    ejes[0, k].set_title(nombre)
    ejes[1, k].imshow(mapa, cmap="hot", vmin=0, vmax=vmax)
    ejes[1, k].set_title("max %.2f" % mx)
for ax in ejes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa9_reales.png", dpi=150)
plt.show()
