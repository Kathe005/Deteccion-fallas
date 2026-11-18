import numpy as np

#una "imsgen" de 4x4 hecha a mano
imagen = np.array([
    [0, 50, 100, 150],
    [50, 100, 150, 200],
    [100, 150, 200, 250],
    [150, 200, 250, 255],
], dtype=np.uint8)

print("Forma (filas, columnas):",imagen.shape)
print("Valor minimo:",imagen.min())
print("Valor maximo:",imagen.max())
print("pixel en fila 2, columna 3:",imagen[2, 3])
print("Primera fila:",imagen[0])
print("Primera columna:",imagen[:, 0])