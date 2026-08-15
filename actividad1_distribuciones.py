"""
Actividad 1: Variables Aleatorias y Distribuciones
Procesamiento de Señales Biomédicas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# ----------------------------------------------------------------------
# 1. Generación de las muestras
# ----------------------------------------------------------------------

np.random.seed(42)  # Para reproducibilidad de resultados
n = 1000

# Normal: podría representar, por ejemplo, frecuencia cardíaca (lpm)
mu_normal, sigma_normal = 100, 15
datos_normal = np.random.normal(loc=mu_normal, scale=sigma_normal, size=n)

# Uniforme: podría representar, por ejemplo, presión arterial diastólica (mmHg)
a_unif, b_unif = 60, 100
datos_uniforme = np.random.uniform(low=a_unif, high=b_unif, size=n)

# Exponencial: podría representar, por ejemplo, tiempos entre eventos (ej. intervalos R-R anómalos)
lambda_exp = 0.05
escala_exp = 1 / lambda_exp  # numpy usa "escala" = 1/lambda
datos_exponencial = np.random.exponential(scale=escala_exp, size=n)


# ----------------------------------------------------------------------
# 2. Histogramas de las tres distribuciones
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(datos_normal, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
axes[0].set_title("Distribución Normal\n(μ=100, σ=15)")
axes[0].set_xlabel("Valor")
axes[0].set_ylabel("Frecuencia")

axes[1].hist(datos_uniforme, bins=30, color="seagreen", edgecolor="black", alpha=0.8)
axes[1].set_title("Distribución Uniforme\n(a=60, b=100)")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frecuencia")

axes[2].hist(datos_exponencial, bins=30, color="indianred", edgecolor="black", alpha=0.8)
axes[2].set_title("Distribución Exponencial\n(λ=0.05)")
axes[2].set_xlabel("Valor")
axes[2].set_ylabel("Frecuencia")

plt.suptitle("Histogramas de Variables Aleatorias (n=1000)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("histogramas_distribuciones.png", dpi=150)
plt.show()


# ----------------------------------------------------------------------
# 3. Comparación de estadísticos muestrales vs poblacionales
# ----------------------------------------------------------------------

# --- Valores poblacionales teóricos ---
# Normal: media = mu, sigma = sigma
media_pob_normal = mu_normal
sigma_pob_normal = sigma_normal

# Uniforme: media = (a+b)/2, sigma = (b-a)/sqrt(12)
media_pob_uniforme = (a_unif + b_unif) / 2
sigma_pob_uniforme = (b_unif - a_unif) / np.sqrt(12)

# Exponencial: media = 1/lambda, sigma = 1/lambda
media_pob_exponencial = 1 / lambda_exp
sigma_pob_exponencial = 1 / lambda_exp

# --- Valores muestrales calculados con las 1000 muestras ---
resultados = pd.DataFrame({
    "Distribución": ["Normal", "Uniforme", "Exponencial"],
    "Media muestral": [
        np.mean(datos_normal),
        np.mean(datos_uniforme),
        np.mean(datos_exponencial),
    ],
    "Media poblacional": [
        media_pob_normal,
        media_pob_uniforme,
        media_pob_exponencial,
    ],
    "Desv. Est. muestral": [
        np.std(datos_normal, ddof=1),
        np.std(datos_uniforme, ddof=1),
        np.std(datos_exponencial, ddof=1),
    ],
    "Desv. Est. poblacional": [
        sigma_pob_normal,
        sigma_pob_uniforme,
        sigma_pob_exponencial,
    ],
})

# Diferencias absolutas para facilitar el análisis
resultados["Dif. Media"] = (resultados["Media muestral"] - resultados["Media poblacional"]).abs()
resultados["Dif. Desv."] = (resultados["Desv. Est. muestral"] - resultados["Desv. Est. poblacional"]).abs()

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n" + "="*80)
print("COMPARACIÓN: ESTADÍSTICOS MUESTRALES vs POBLACIONALES (n=1000)")
print("="*80)
print(resultados.to_string(index=False))
print("="*80)


# ----------------------------------------------------------------------
# 4. Interpretación: ¿qué señal biomédica modelaría cada distribución?
# ----------------------------------------------------------------------

interpretacion = """
INTERPRETACIÓN - APLICACIÓN A SEÑALES BIOMÉDICAS
--------------------------------------------------
1. Distribución NORMAL (μ=100, σ=15):
   Es apropiada para variables fisiológicas que fluctúan simétricamente
   alrededor de un valor central por múltiples factores independientes,
   como la FRECUENCIA CARDÍACA en reposo, la GLUCEMIA en una población
   sana, o la PRESIÓN ARTERIAL SISTÓLICA. El teorema del límite central
   respalda su uso cuando la variabilidad es resultado de muchas
   pequeñas causas aleatorias sumadas.

2. Distribución UNIFORME (60-100):
   Es poco común como modelo fisiológico "natural", ya que la mayoría
   de señales biomédicas tienden a concentrarse alrededor de un valor
   típico. Sin embargo, puede ser útil para modelar RUIDO DE CUANTIZACIÓN
   en un conversor analógico-digital, o el TIEMPO DE LLEGADA de eventos
   dentro de una ventana de muestreo, donde no hay razón para que un
   valor sea más probable que otro dentro de un rango acotado.

3. Distribución EXPONENCIAL (λ=0.05):
   Es ideal para modelar TIEMPOS ENTRE EVENTOS aleatorios que ocurren
   a una tasa constante, como el TIEMPO ENTRE LATIDOS ANÓMALOS o
   EXTRASÍSTOLES, el TIEMPO ENTRE DISPAROS NEURONALES (spikes) en
   registros de electrofisiología, o el TIEMPO DE VIDA de componentes
   en equipos de monitoreo. Su asimetría (cola larga a la derecha)
   refleja que eventos muy espaciados en el tiempo son posibles pero
   poco frecuentes.

En resumen: la elección de la distribución depende de la naturaleza
del fenómeno biomédico —si es una medición fisiológica continua con
tendencia central (Normal), un proceso sin preferencia dentro de un
rango (Uniforme), o un proceso de eventos discretos en el tiempo
(Exponencial).
"""

print(interpretacion)

# Guardar resultados en CSV para el informe
resultados.to_csv("resultados_estadisticos.csv", index=False)
print("\nArchivos generados: 'histogramas_distribuciones.png' y 'resultados_estadisticos.csv'")
