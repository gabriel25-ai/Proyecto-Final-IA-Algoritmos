import numpy as np

# --- PARTE A: REGRESIÓN LINEAL SIMPLE ---
def calcular_regresion(x, y):
    n = len(x)
    # Cálculo manual de medias
    media_x = sum(x) / n
    media_y = sum(y) / n
    
    # Cálculo de pendiente (m) e intercepto (b) usando Mínimos Cuadrados
    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = sum((x[i] - media_x)**2 for i in range(n))
    
    m = numerador / denominador
    b = media_y - (m * media_x)
    
    # Cálculo del Error Cuadrático Medio (MSE)
    predicciones = [m * xi + b for xi in x]
    mse = sum((y[i] - predicciones[i])**2 for i in range(n)) / n
    
    return m, b, mse

# --- PARTE B: K-NEAREST NEIGHBORS (KNN) ---
def distancia_euclidiana(punto1, punto2):
    # Fórmula: sqrt(sum((pi - qi)^2))
    return np.sqrt(sum((px - qx) ** 2 for px, qx in zip(punto1, punto2)))

def clasificar_knn(data_x, data_y, nuevo_punto, k):
    distancias = []
    for i in range(len(data_x)):
        dist = distancia_euclidiana(data_x[i], nuevo_punto)
        distancias.append((dist, data_y[i]))
    
    # Ordenar por distancia y tomar los K vecinos más cercanos
    distancias.sort(key=lambda x: x[0])
    vecinos = distancias[:k]
    
    # Votación por mayoría
    votos = [v[1] for v in vecinos]
    return max(set(votos), key=votos.count)