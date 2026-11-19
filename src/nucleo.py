import numpy as np

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
    """Un solo numero por imagen: log del MSE maximo por bloque."""
    return np.log(1 + mse_por_bloques(img, patron).max())

def pieza_mala(patron):
    lado = np.random.randint(3, 9)
    fila = np.random.randint(4, 64 - lado - 4)
    col = np.random.randint(4, 64 - lado - 4)
    valor = np.random.choice([20, 40, 60])
    return agregar_ruido(agregar_defecto(patron, fila, col, lado, valor))