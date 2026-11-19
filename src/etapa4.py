import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

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

def agregar_defecto(img, fila, col, lado, valor):
    salida = img.copy()
    salida[fila:fila + lado, col:col + lado] = valor
    return salida

def mse_por_bloques(prueba, patron, bloque=8):
    filas = prueba.shape[0] // bloque
    cols = prueba.shape[1] // bloque
    mapa = np.zeros((filas, cols))
    for i in range(filas):
        for j in range(cols):
            f0, f1 = i * bloque, (i + 1) * bloque
            c0, c1 = j * bloque, (j + 1) * bloque
            dif = prueba[f0:f1, c0:c1] - patron[f0:f1, c0:c1]
            mapa[i, j] = np.mean(dif ** 2)
    return mapa

def caracteristica(img, patron):
    return mse_por_bloques(img, patron).max()

patron = crear_patron()

# Conjunto de ejemplos: 100 buenas y 100 malas
car_buenas = []
car_malas = []
for _ in range(100):
    car_buenas.append(caracteristica(agregar_ruido(patron), patron))
    lado = np.random.randint(3, 9)
    fila = np.random.randint(4, 64 - lado - 4)
    col = np.random.randint(4, 64 - lado - 4)
    valor = np.random.choice([20, 40, 60])
    mala = agregar_ruido(agregar_defecto(patron, fila, col, lado, valor))
    car_malas.append(caracteristica(mala, patron))

car_buenas = np.array(car_buenas)
car_malas = np.array(car_malas)

# Buscar el umbral con mayor exactitud
candidatos = np.linspace(0, car_malas.max(), 500)
mejor_umbral, mejor_exactitud = 0, 0
for u in candidatos:
    aciertos = np.sum(car_buenas < u) + np.sum(car_malas >= u)
    exactitud = aciertos / 200
    if exactitud > mejor_exactitud:
        mejor_exactitud, mejor_umbral = exactitud, u

print("Buenas: min %.1f  max %.1f" % (car_buenas.min(), car_buenas.max()))
print("Malas:  min %.1f  max %.1f" % (car_malas.min(), car_malas.max()))
print("Umbral aprendido:", round(mejor_umbral, 2))
print("Exactitud:", mejor_exactitud)

plt.hist(car_buenas, bins=20, alpha=0.7, label="Buenas")
plt.hist(car_malas, bins=20, alpha=0.7, label="Malas")
plt.axvline(mejor_umbral, color="red", linestyle="--", label="Umbral")
plt.xlabel("MSE maximo por bloque")
plt.ylabel("Cantidad de piezas")
plt.legend()
os.makedirs("resultados", exist_ok=True)
plt.savefig("resultados/etapa2_diferencia.png", dpi=150)
plt.show()