import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================================================
# --- PARTE 1: LÓGICA MATEMÁTICA (Implementación Manual) ---
# =========================================================

def calcular_regresion_manual(x, y):
    n = len(x)
    if n == 0: return 0, 0, 0
    media_x = sum(x) / n
    media_y = sum(y) / n
    numerador = sum((x[i] - media_x) * (y[i] - media_y) for i in range(n))
    denominador = sum((x[i] - media_x)**2 for i in range(n))
    m = numerador / denominador if denominador != 0 else 0
    b = media_y - (m * media_x)
    predicciones = [m * xi + b for xi in x]
    mse = sum((y[i] - predicciones[i])**2 for i in range(n)) / n
    return m, b, mse

def distancia_euclidiana_manual(punto1, punto2):
    return np.sqrt(sum((float(px) - float(qx)) ** 2 for px, qx in zip(punto1, punto2)))

def clasificar_knn_manual(data_x, data_y, nuevo_punto, k):
    distancias = []
    for i in range(len(data_x)):
        dist = distancia_euclidiana_manual(data_x[i], nuevo_punto)
        distancias.append((dist, data_y[i]))
    distancias.sort(key=lambda x: x[0])
    vecinos = distancias[:k]
    votos = [v[1] for v in vecinos]
    return max(set(votos), key=votos.count)

# =========================================================
# --- PARTE 2: INTERFAZ GRÁFICA (GUI) ---
# =========================================================

class AppIA(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Proyecto Final IA - Unificado")
        self.geometry("1100x750")
        self.df = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel Lateral
        self.sidebar = ctk.CTkFrame(self, width=280)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkButton(self.sidebar, text="1. Cargar CSV", command=self.cargar_archivo).pack(pady=20, padx=20)

        # Sección Regresión
        ctk.CTkLabel(self.sidebar, text="REGRESIÓN LINEAL", font=("Arial", 14, "bold")).pack(pady=(10,5))
        self.entry_x = ctk.CTkEntry(self.sidebar, placeholder_text="Valor X para predecir")
        self.entry_x.pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Calcular Regresión", command=self.ejecutar_regresion).pack(pady=5, padx=20)

        # Sección KNN
        ctk.CTkLabel(self.sidebar, text="K-NN CLASIFICACIÓN", font=("Arial", 14, "bold")).pack(pady=(20,5))
        self.entry_k = ctk.CTkEntry(self.sidebar, placeholder_text="Valor de K (ej: 3)")
        self.entry_k.pack(pady=5, padx=20)
        self.entry_knn_punto = ctk.CTkEntry(self.sidebar, placeholder_text="Punto X,Y (ej: 4,5)")
        self.entry_knn_punto.pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Clasificar con KNN", command=self.ejecutar_knn).pack(pady=5, padx=20)

        # Panel Central
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.lbl_res = ctk.CTkLabel(self.main_frame, text="Resultados aparecerán aquí", font=("Arial", 13), justify="left")
        self.lbl_res.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def cargar_archivo(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path, sep=';').dropna()
                messagebox.showinfo("Éxito", "Archivo cargado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al leer el CSV: {e}")

    def ejecutar_regresion(self):
        if self.df is None: 
            messagebox.showerror("Error", "Carga el CSV primero")
            return
        try:
            x = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').dropna().values
            y = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').dropna().values
            m, b, mse = calcular_regresion_manual(x, y)
            
            val_x = self.entry_x.get()
            pred_txt = ""
            if val_x:
                y_pred = (m * float(val_x)) + b
                pred_txt = f"\nPredicción: Para X={val_x}, Y={y_pred:.2f}"
            
            self.lbl_res.configure(text=f"Ecuación: Y = {m:.2f}X + {b:.2f}\nMSE: {mse:.4f}{pred_txt}")
            
            self.ax.clear()
            self.ax.scatter(x, y, color="blue", label="Datos")
            l_x = np.array([min(x), max(x)])
            self.ax.plot(l_x, m*l_x + b, color="red", label="Regresión")
            self.ax.set_title("Regresión Lineal Simple")
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error en regresión: {e}")

    def ejecutar_knn(self):
        if self.df is None: 
            messagebox.showerror("Error", "Carga el CSV primero")
            return
        try:
            k = int(self.entry_k.get())
            punto_txt = self.entry_knn_punto.get()
            
            if not punto_txt:
                punto_nuevo = [self.df.iloc[:, 0].mean(), self.df.iloc[:, 1].mean()]
            else:
                punto_nuevo = [float(i) for i in punto_txt.split(',')]
            
            data_x = self.df.iloc[:, 0:2].values.astype(float)
            data_y = self.df.iloc[:, 2].values
            
            clase = clasificar_knn_manual(data_x, data_y, punto_nuevo, k)
            self.lbl_res.configure(text=f"Resultado KNN: El punto {punto_nuevo} es Clase: {clase}")
            
            self.ax.clear()
            colores = pd.factorize(data_y)[0]
            self.ax.scatter(data_x[:, 0], data_x[:, 1], c=colores, cmap='viridis')
            self.ax.scatter(punto_nuevo[0], punto_nuevo[1], color='red', marker='x', s=200, label="Nuevo Punto")
            self.ax.set_title(f"Clasificación KNN (K={k})")
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Verifica K y el punto: {e}")

if __name__ == "__main__":
    app = AppIA()
    app.mainloop()