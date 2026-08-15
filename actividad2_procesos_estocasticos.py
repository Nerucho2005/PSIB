"""
Actividad 2: Simulación de Procesos Estocásticos
Procesamiento de Señales Biomédicas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

np.random.seed(42)

# ----------------------------------------------------------------------
# Parámetros del proceso
# ----------------------------------------------------------------------
A = 2                 # Amplitud fija
f0 = 5                # Frecuencia fija (Hz)
n_realizaciones = 50
n_puntos = 300
t = np.linspace(0, 3, n_puntos)   # Vector de tiempo [0, 3] s

# ----------------------------------------------------------------------
# 1. Generar 50 realizaciones de X(t) = A*cos(2*pi*f0*t + Phi)
#    con Phi ~ Uniforme[0, 2*pi] (una fase distinta por realización)
# ----------------------------------------------------------------------

phi = np.random.uniform(0, 2*np.pi, size=n_realizaciones)  # 50 fases aleatorias

# Matriz de realizaciones: filas = realizaciones, columnas = instantes de tiempo
X = np.array([A * np.cos(2*np.pi*f0*t + phi[i]) for i in range(n_realizaciones)])
# X tiene forma (50, 300)

# ----------------------------------------------------------------------
# 2. Graficar las primeras 10 realizaciones
# ----------------------------------------------------------------------

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.plot(t, X[i], alpha=0.8, label=f"Realización {i+1}" if i < 5 else None)
plt.title("Primeras 10 realizaciones de X(t) = A·cos(2πf₀t + Φ)")
plt.xlabel("Tiempo (s)")
plt.ylabel("X(t)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("proceso_X_realizaciones.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 3. Media empírica μ̂(t): promedio sobre las 50 realizaciones, para cada t
# ----------------------------------------------------------------------

media_empirica_X = np.mean(X, axis=0)   # promedio a lo largo de las realizaciones (eje 0)

# Media teórica: E[X(t)] = A * E[cos(2*pi*f0*t + Phi)] = 0 para todo t
# (porque al integrar cos sobre una fase uniforme en [0, 2*pi] el resultado es 0)
media_teorica_X = np.zeros_like(t)

# ----------------------------------------------------------------------
# 4. Graficar media empírica vs media teórica
# ----------------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(t, media_empirica_X, color="steelblue", label="Media empírica μ̂(t)")
plt.plot(t, media_teorica_X, color="red", linestyle="--", label="Media teórica μ(t) = 0")
plt.title("Media empírica vs media teórica — Proceso X(t)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Media")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("proceso_X_media.png", dpi=150)
plt.show()

print("="*70)
print("PROCESO X(t) = A·cos(2πf₀t + Φ)")
print("="*70)
print(f"Media empírica promedio (sobre todo t): {np.mean(media_empirica_X):.5f}")
print(f"Máxima desviación |μ̂(t) - μ_teórica(t)|: {np.max(np.abs(media_empirica_X - media_teorica_X)):.5f}")


# ----------------------------------------------------------------------
# 5. ¿Es estacionario? -> Ver interpretación al final del script
# ----------------------------------------------------------------------


# ========================================================================
# SEGUNDA PARTE: Proceso con deriva lineal
# Y(t) = X(t) + 0.5*t
# ========================================================================

deriva = 0.5 * t                       # 0.5*t, misma para todas las realizaciones
Y = X + deriva                         # se suma la deriva a cada realización (broadcasting)

# --- Graficar primeras 10 realizaciones de Y(t) ---
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.plot(t, Y[i], alpha=0.8)
plt.plot(t, deriva, color="black", linewidth=2, linestyle="--", label="Deriva 0.5t")
plt.title("Primeras 10 realizaciones de Y(t) = X(t) + 0.5t")
plt.xlabel("Tiempo (s)")
plt.ylabel("Y(t)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("proceso_Y_realizaciones.png", dpi=150)
plt.show()

# --- Media empírica y teórica de Y(t) ---
media_empirica_Y = np.mean(Y, axis=0)
# E[Y(t)] = E[X(t)] + 0.5t = 0 + 0.5t = 0.5t
media_teorica_Y = 0.5 * t

plt.figure(figsize=(12, 5))
plt.plot(t, media_empirica_Y, color="steelblue", label="Media empírica μ̂_Y(t)")
plt.plot(t, media_teorica_Y, color="red", linestyle="--", label="Media teórica μ_Y(t) = 0.5t")
plt.title("Media empírica vs media teórica — Proceso Y(t) = X(t) + 0.5t")
plt.xlabel("Tiempo (s)")
plt.ylabel("Media")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("proceso_Y_media.png", dpi=150)
plt.show()

print("\n" + "="*70)
print("PROCESO Y(t) = X(t) + 0.5t")
print("="*70)
print(f"Media empírica en t=0: {media_empirica_Y[0]:.5f}  (teórica: {media_teorica_Y[0]:.5f})")
print(f"Media empírica en t=3: {media_empirica_Y[-1]:.5f}  (teórica: {media_teorica_Y[-1]:.5f})")
print(f"Máxima desviación |μ̂_Y(t) - μ_teórica_Y(t)|: {np.max(np.abs(media_empirica_Y - media_teorica_Y)):.5f}")


# ----------------------------------------------------------------------
# Comparación conjunta de ambas medias
# ----------------------------------------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(t, media_empirica_X, label="μ̂(t) de X(t)  (sin deriva)", color="steelblue")
plt.plot(t, media_empirica_Y, label="μ̂(t) de Y(t)  (con deriva)", color="darkorange")
plt.axhline(0, color="gray", linestyle=":", linewidth=1)
plt.title("Comparación de medias empíricas: X(t) vs Y(t)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Media")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_medias_X_Y.png", dpi=150)
plt.show()


# ----------------------------------------------------------------------
# INTERPRETACIÓN / RESPUESTAS
# ----------------------------------------------------------------------

interpretacion = """
================================================================================
INTERPRETACIÓN
================================================================================

¿ES EL PROCESO X(t) = A·cos(2πf₀t + Φ) ESTACIONARIO?

Sí, X(t) es un proceso estacionario en sentido amplio (WSS, Wide-Sense
Stationary). Esto se debe a que:

1. MEDIA CONSTANTE EN EL TIEMPO:
   E[X(t)] = A · E[cos(2πf₀t + Φ)] = 0 para todo t, ya que Φ está
   distribuida uniformemente en [0, 2π], y al promediar el coseno sobre
   una fase uniforme completa el resultado es cero, independientemente
   del valor de t. La media empírica μ̂(t) obtenida en la simulación
   oscila alrededor de 0 (con pequeñas fluctuaciones por ser una
   estimación con solo 50 realizaciones), lo cual confirma esto.

2. AUTOCORRELACIÓN QUE DEPENDE SOLO DEL DESFASE (τ = t2 - t1):
   Puede demostrarse que R_X(t1, t2) = (A²/2)·cos(2πf₀(t2 - t1)), es
   decir, depende únicamente de la diferencia de tiempos τ = t2 - t1 y
   no de los valores absolutos de t1 y t2.

Al cumplirse ambas condiciones (media constante y autocorrelación
dependiente solo del desfase), el proceso es estacionario en sentido
amplio. Nota: la fase aleatoria Φ es justamente lo que "distribuye" la
variabilidad de forma homogénea en el tiempo, haciendo que las
propiedades estadísticas del proceso no cambien conforme avanza t.


¿QUÉ SUCEDE CON Y(t) = X(t) + 0.5t?

Y(t) NO es estacionario. Al sumar la deriva lineal 0.5t, la media deja
de ser constante:

   E[Y(t)] = E[X(t)] + 0.5t = 0 + 0.5t = 0.5t

Es decir, la media teórica ahora depende explícitamente de t (crece
linealmente). Esto se confirma en la simulación: la media empírica de
Y(t) pasa de estar cerca de 0 en t=0 a un valor cercano a 1.5 en t=3
(0.5 × 3), siguiendo la línea de tendencia 0.5t.

DIFERENCIA CLAVE:
- X(t): media empírica ≈ constante (≈0) para todo t → proceso
  estacionario.
- Y(t): media empírica crece linealmente con t, siguiendo la deriva
  0.5t → proceso NO estacionario (la media varía con el tiempo).

En términos de señales biomédicas, esto es análogo a comparar una señal
oscilatoria "limpia" (por ejemplo, un tono de referencia o un ritmo
cardíaco estable) con una señal que presenta una TENDENCIA o DERIVA
(drift), como ocurre en registros de ECG o EEG afectados por el
movimiento del electrodo, la respiración o cambios de impedancia de
piel, donde la línea base ya no es constante. En procesamiento de
señales biomédicas, detectar y remover ese tipo de deriva (por ejemplo,
con filtros pasa-altos o detrending) es un paso habitual de
preprocesamiento antes de analizar la señal.
================================================================================
"""

print(interpretacion)

# Guardar tabla resumen
resumen = pd.DataFrame({
    "t": t,
    "media_empirica_X": media_empirica_X,
    "media_teorica_X": media_teorica_X,
    "media_empirica_Y": media_empirica_Y,
    "media_teorica_Y": media_teorica_Y,
})
resumen.to_csv("resumen_procesos_X_Y.csv", index=False)
print("Archivos generados: proceso_X_realizaciones.png, proceso_X_media.png,")
print("proceso_Y_realizaciones.png, proceso_Y_media.png, comparacion_medias_X_Y.png,")
print("resumen_procesos_X_Y.csv")
