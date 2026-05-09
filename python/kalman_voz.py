"""
kalman_voz.py
=============
Funciones auxiliares para el análisis de filtro de Kalman
aplicado a señales de voz sintéticas y reconstrucción con
distintos métodos (polinomio, spline, Kalman).
"""

import numpy as np


# ── Generación de señal ──────────────────────────────────────────────────────

def generar_senal_voz(n_muestras=400, fs=8000.0, freq_fundamental=150.0, seed=42):
    """
    Genera una señal de voz sintética (suma de armónicos) con envolvente.

    Returns:
        t: array de tiempos (segundos)
        x: señal limpia
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_muestras) / fs

    # Fundamental + armónicos
    x = np.zeros(n_muestras)
    for k in range(1, 6):  # 5 armónicos
        amp = 1.0 / k
        fase = rng.uniform(0, 2 * np.pi)
        x += amp * np.sin(2 * np.pi * freq_fundamental * k * t + fase)

    # Envolvente suave
    envolvente = np.sin(np.pi * np.arange(n_muestras) / n_muestras) ** 0.5
    x *= envolvente

    # Normalizar a [-1, 1]
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))

    return t, x


def agregar_ruido(x, snr_db=10.0, seed=0):
    """
    Agrega ruido blanco gaussiano a la señal x con una SNR dada en dB.
    """
    rng = np.random.RandomState(seed)
    potencia_senal = np.mean(x ** 2)
    potencia_ruido = potencia_senal / (10 ** (snr_db / 10))
    ruido = rng.normal(0, np.sqrt(potencia_ruido), len(x))
    return x + ruido


# ── Filtro de Kalman ─────────────────────────────────────────────────────────

def filtro_kalman(z, Q=0.01, R=0.1, x0=0.0, P0=1.0):
    """
    Filtro de Kalman 1D escalar.

    Args:
        z: señal observada (puede contener NaN para datos perdidos)
        Q: varianza del ruido de proceso
        R: varianza del ruido de medición
        x0: estado inicial
        P0: covarianza inicial

    Returns:
        x_est: estimación filtrada
        ganancias: ganancia K en cada paso
        covarianzas: covarianza P en cada paso
    """
    n = len(z)
    x_est = np.zeros(n)
    ganancias = np.zeros(n)
    covarianzas = np.zeros(n)

    x = x0
    P = P0

    for i in range(n):
        # Predicción
        x_pred = x
        P_pred = P + Q

        if not np.isnan(z[i]):
            # Corrección
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z[i] - x_pred)
            P = (1 - K) * P_pred
        else:
            K = 0.0
            x = x_pred
            P = P_pred

        x_est[i] = x
        ganancias[i] = K
        covarianzas[i] = P

    return x_est, ganancias, covarianzas


# ── Métodos de interpolación ─────────────────────────────────────────────────

def interpolar_polinomio(t, z, grado=8):
    """Ajuste polinomial por mínimos cuadrados."""
    coefs = np.polyfit(t, z, grado)
    return np.polyval(coefs, t)


def interpolar_spline(t, z, factor=5):
    """Interpolación con spline cúbico submuestreado."""
    from scipy.interpolate import CubicSpline
    indices = np.arange(0, len(t), factor)
    cs = CubicSpline(t[indices], z[indices])
    return cs(t)


# ── Métricas ─────────────────────────────────────────────────────────────────

def calcular_mse(original, estimada):
    """Error cuadrático medio."""
    return float(np.mean((original - estimada) ** 2))


def calcular_snr(original, estimada):
    """Relación señal/ruido en dB."""
    error = original - estimada
    pot_senal = np.mean(original ** 2)
    pot_error = np.mean(error ** 2)
    if pot_error == 0:
        return float('inf')
    return float(10 * np.log10(pot_senal / pot_error))


# ── Comparación de métodos ───────────────────────────────────────────────────

def comparar_metodos(t, x_original, z_ruidosa, grado_poly=8, factor_spline=5,
                     Q=0.01, R=0.01):
    """
    Compara polinomio, spline y Kalman contra la señal original.

    Returns:
        dict con claves: polinomio, spline, kalman, metricas
    """
    # Polinomio
    x_poly = interpolar_polinomio(t, z_ruidosa, grado=grado_poly)

    # Spline
    x_spline = interpolar_spline(t, z_ruidosa, factor=factor_spline)

    # Kalman
    x_kalman, _, _ = filtro_kalman(z_ruidosa, Q=Q, R=R)

    metricas = {
        "Ruidosa": {
            "mse": calcular_mse(x_original, z_ruidosa),
            "snr": calcular_snr(x_original, z_ruidosa),
        },
        "Polinomio": {
            "mse": calcular_mse(x_original, x_poly),
            "snr": calcular_snr(x_original, x_poly),
        },
        "Spline cúbico": {
            "mse": calcular_mse(x_original, x_spline),
            "snr": calcular_snr(x_original, x_spline),
        },
        "Kalman": {
            "mse": calcular_mse(x_original, x_kalman),
            "snr": calcular_snr(x_original, x_kalman),
        },
    }

    return {
        "polinomio": x_poly,
        "spline": x_spline,
        "kalman": x_kalman,
        "metricas": metricas,
    }