"""
kalman_analisis_completo.py
===========================
Análisis completo del filtro de Kalman:
  Parte A — Script proporcionado (señal senoidal con pérdida de paquetes)
  Parte B — Caso de estudio avanzado (voz sintética con ruido, comparación de métodos)

Genera 6 figuras PNG en la carpeta de salida.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Asegurar que encontremos kalman_voz.py
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from kalman_voz import (
    generar_senal_voz,
    agregar_ruido,
    comparar_metodos,
    calcular_mse,
    calcular_snr,
    filtro_kalman,
    interpolar_polinomio,
    interpolar_spline,
)

# ── Directorio de salida ──────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(_dir), "scratch")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#475569",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "font.size":        11,
})

# ══════════════════════════════════════════════════════════════════════════════
# PARTE A — Script proporcionado: pérdida de paquetes en señal senoidal
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  PARTE A — Filtro de Kalman con pérdida de paquetes (señal senoidal)")
print("=" * 70)

# Señal original
t = np.linspace(0, 10, 100)
signal = np.sin(t)

# Simular pérdida de datos (35% disponible, 65% perdido)
np.random.seed(0)
mask = np.random.rand(len(signal)) > 0.65
received = signal.copy()
received[mask] = np.nan

n_perdidos = int(np.sum(mask))
n_disponibles = len(signal) - n_perdidos
print(f"\n  Muestras totales:     {len(signal)}")
print(f"  Muestras recibidas:   {n_disponibles} ({100*n_disponibles/len(signal):.0f}%)")
print(f"  Muestras perdidas:    {n_perdidos} ({100*n_perdidos/len(signal):.0f}%)")

# Parámetros Kalman
Q = 0.01   # varianza del ruido de proceso
R = 0.1    # varianza del ruido de medición

x = 0.0    # estado inicial
P = 0.101  # covarianza inicial

recon = []
ganancias = []
covarianzas = []

for z in received:
    # Predicción
    x_pred = x
    P_pred = P + Q

    if not np.isnan(z):
        # Corrección — HAY medición
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
    else:
        # Sin dato — solo predicción
        K = 0.0
        x = x_pred
        P = P_pred

    recon.append(x)
    ganancias.append(K)
    covarianzas.append(P)

recon = np.array(recon)

# Calcular error donde NO hay pérdida (muestras válidas)
validos = ~mask
mse_recon = np.mean((signal[validos] - recon[validos])**2)
mse_total = np.mean((signal - recon)**2)
print(f"\n  MSE (muestras recibidas): {mse_recon:.6f}")
print(f"  MSE (total):              {mse_total:.6f}")

# ── Figura 1: Gráfica principal del script ────────────────────────────────────
fig1, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, signal, color="#60a5fa", linewidth=1.5, alpha=0.7, label="Original — sin(t)")
ax.plot(t, received, 'o', color="#f97316", markersize=5, alpha=0.8, label=f"Recibida ({n_disponibles} muestras)")
ax.plot(t, recon, color="#22c55e", linewidth=2.0, label="Reconstrucción Kalman")

# Sombrear zonas de pérdida
for i in range(len(mask)):
    if mask[i]:
        ax.axvspan(t[max(0,i-1)], t[min(len(t)-1,i)], alpha=0.07, color="#ef4444")

ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
ax.set_title("Parte A — Reconstrucción con Pérdida de Paquetes (65% pérdida)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT, "fig1_senoidal_kalman.png"), dpi=150, bbox_inches="tight")
print(f"\n  ✓ fig1_senoidal_kalman.png guardada")

# ── Figura 2: Ganancia de Kalman y Covarianza ─────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(t, ganancias, color="#a78bfa", linewidth=1.2)
ax1.fill_between(t, 0, ganancias, alpha=0.2, color="#a78bfa")
ax1.set_ylabel("Ganancia K")
ax1.set_title("Evolución de la Ganancia de Kalman", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)
# Marcar donde K=0 (pérdida)
for i in range(len(ganancias)):
    if ganancias[i] == 0:
        ax1.axvline(t[i], color="#ef4444", alpha=0.15, linewidth=1)

ax2.plot(t, covarianzas, color="#fb923c", linewidth=1.2)
ax2.fill_between(t, 0, covarianzas, alpha=0.2, color="#fb923c")
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Covarianza P")
ax2.set_title("Evolución de la Covarianza del Error", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "fig2_ganancia_covarianza.png"), dpi=150, bbox_inches="tight")
print("  ✓ fig2_ganancia_covarianza.png guardada")

# ── Figura 3: Error puntual ──────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(12, 4))
error = np.abs(signal - recon)
colors = ["#ef4444" if mask[i] else "#22c55e" for i in range(len(error))]
ax.bar(t, error, width=0.08, color=colors, alpha=0.8)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("|Error|")
ax.set_title("Error Absoluto Puntual (rojo = dato perdido, verde = dato recibido)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT, "fig3_error_puntual.png"), dpi=150, bbox_inches="tight")
print("  ✓ fig3_error_puntual.png guardada")


# ══════════════════════════════════════════════════════════════════════════════
# PARTE B — Caso de Estudio: Señal de voz sintética con ruido
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PARTE B — Caso de Estudio: Reconstrucción de señal de voz")
print("=" * 70)

# Generar señal de voz sintética
n_muestras = 400
fs = 8000.0
f0 = 150.0
snr_entrada_db = 10.0

t_voz, x_original = generar_senal_voz(n_muestras=n_muestras, fs=fs, freq_fundamental=f0, seed=42)
z_ruidosa = agregar_ruido(x_original, snr_db=snr_entrada_db, seed=0)

# Comparar métodos
Q_voz = 1e-2
R_voz = 1e-2
grado_poly = 8
factor_spline = 5

res = comparar_metodos(
    t_voz, x_original, z_ruidosa,
    grado_poly=grado_poly,
    factor_spline=factor_spline,
    Q=Q_voz, R=R_voz,
)
metricas = res["metricas"]

print(f"\n  Configuración:")
print(f"    N = {n_muestras}, fs = {fs} Hz, f0 = {f0} Hz")
print(f"    SNR entrada = {snr_entrada_db} dB")
print(f"    Kalman: Q = {Q_voz}, R = {R_voz}")
print(f"    Polinomio: grado = {grado_poly}")
print(f"    Spline: submuestreo = {factor_spline}")
print(f"\n  {'Método':<25} {'MSE':>12} {'SNR (dB)':>12}")
print(f"  {'─'*25} {'─'*12} {'─'*12}")
for nombre, m in metricas.items():
    print(f"  {nombre:<25} {m['mse']:>12.6f} {m['snr']:>12.2f}")

# ── Figura 4: Comparación de métodos (señal de voz) ──────────────────────────
t_ms = t_voz * 1000  # convertir a ms

fig4, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel superior: Original + Ruidosa
axes[0].plot(t_ms, x_original, color="#60a5fa", linewidth=1.0, alpha=0.8, label="Original")
axes[0].plot(t_ms, z_ruidosa, color="#94a3b8", linewidth=0.5, alpha=0.5, label="Ruidosa (SNR=10dB)")
axes[0].set_ylabel("Amplitud")
axes[0].set_title("Señal de Voz — Original vs Ruidosa", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# Panel medio: Polinomio y Spline
axes[1].plot(t_ms, x_original, color="#60a5fa", linewidth=1.0, alpha=0.4, label="Original")
axes[1].plot(t_ms, res["polinomio"], color="#ef4444", linewidth=1.2, alpha=0.8,
             label=f"Polinomio (MSE={metricas['Polinomio']['mse']:.4f})")
axes[1].plot(t_ms, res["spline"], color="#eab308", linewidth=1.2, alpha=0.8,
             label=f"Spline (MSE={metricas['Spline cúbico']['mse']:.4f})")
axes[1].set_ylabel("Amplitud")
axes[1].set_title("Métodos Clásicos — Polinomio y Spline", fontsize=13, fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

# Panel inferior: Kalman
axes[2].plot(t_ms, x_original, color="#60a5fa", linewidth=1.0, alpha=0.4, label="Original")
axes[2].plot(t_ms, res["kalman"], color="#22c55e", linewidth=1.5, alpha=0.9,
             label=f"Kalman (MSE={metricas['Kalman']['mse']:.4f})")
axes[2].set_xlabel("Tiempo (ms)")
axes[2].set_ylabel("Amplitud")
axes[2].set_title("Filtro de Kalman — Estimación Óptima", fontsize=13, fontweight="bold")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

fig4.suptitle("Caso de Estudio — Reconstrucción de Señal de Voz", fontsize=15, fontweight="bold", y=1.01)
fig4.tight_layout()
fig4.savefig(os.path.join(OUT, "fig4_voz_comparacion.png"), dpi=150, bbox_inches="tight")
print(f"\n  ✓ fig4_voz_comparacion.png guardada")

# ── Figura 5: Error absoluto por método ───────────────────────────────────────
fig5, ax = plt.subplots(figsize=(14, 5))

err_ruidosa  = np.abs(x_original - z_ruidosa)
err_poly     = np.abs(x_original - res["polinomio"])
err_spline   = np.abs(x_original - res["spline"])
err_kalman   = np.abs(x_original - res["kalman"])

ax.plot(t_ms, err_ruidosa, color="#94a3b8", linewidth=0.5, alpha=0.4, label="Señal ruidosa")
ax.plot(t_ms, err_poly,    color="#ef4444", linewidth=0.8, alpha=0.6, label="Polinomio")
ax.plot(t_ms, err_spline,  color="#eab308", linewidth=0.8, alpha=0.6, label="Spline cúbico")
ax.plot(t_ms, err_kalman,  color="#22c55e", linewidth=1.2, alpha=0.9, label="Kalman")
ax.set_xlabel("Tiempo (ms)")
ax.set_ylabel("|Error|")
ax.set_title("Error Absoluto — Comparación de Métodos", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT, "fig5_error_comparacion.png"), dpi=150, bbox_inches="tight")
print("  ✓ fig5_error_comparacion.png guardada")

# ── Figura 6: Tabla de métricas como gráfico de barras ────────────────────────
fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

nombres = list(metricas.keys())
mse_vals = [m["mse"] for m in metricas.values()]
snr_vals = [m["snr"] for m in metricas.values()]
colores = ["#94a3b8", "#ef4444", "#eab308", "#22c55e"]

bars1 = ax1.bar(nombres, mse_vals, color=colores, edgecolor="#1e293b", linewidth=1.5)
ax1.set_ylabel("MSE")
ax1.set_title("Error Cuadrático Medio (menor = mejor)", fontsize=12, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars1, mse_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

bars2 = ax2.bar(nombres, snr_vals, color=colores, edgecolor="#1e293b", linewidth=1.5)
ax2.set_ylabel("SNR (dB)")
ax2.set_title("Relación Señal/Ruido (mayor = mejor)", fontsize=12, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars2, snr_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

fig6.suptitle("Métricas de Reconstrucción — Caso de Estudio Voz", fontsize=14, fontweight="bold")
fig6.tight_layout()
fig6.savefig(os.path.join(OUT, "fig6_metricas_barras.png"), dpi=150, bbox_inches="tight")
print("  ✓ fig6_metricas_barras.png guardada")

print(f"\n  Todas las figuras guardadas en: {OUT}/")
print("=" * 70)
