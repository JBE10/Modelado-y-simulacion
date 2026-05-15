"""filtro_kalman_module.py — Filtro de Kalman (Voz) UI Module."""
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from kalman_voz import (
    generar_senal_voz, agregar_ruido, comparar_metodos, calcular_mse, calcular_snr, filtro_kalman,
)


class KalmanModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Filtro de Kalman (Voz)"

    def render(self, **kwargs):
        def _formulas_kalman():
            with st.container(border=True):
                st.markdown("### Modelo de Estado")
                st.latex(r"x_k = x_{k-1} + w_k, \quad w_k \sim \mathcal{N}(0,Q)")
                st.latex(r"z_k = x_k + v_k, \quad v_k \sim \mathcal{N}(0,R)")
                st.divider()
                st.markdown("### Predicción")
                st.latex(r"\hat{x}_{k|k-1} = \hat{x}_{k-1|k-1}")
                st.latex(r"P_{k|k-1} = P_{k-1|k-1} + Q")
                st.divider()
                st.markdown("### Corrección")
                st.latex(r"K_k = \frac{P_{k|k-1}}{P_{k|k-1} + R}")
                st.latex(r"\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k\,(z_k - \hat{x}_{k|k-1})")
                st.latex(r"P_{k|k} = (1-K_k)\,P_{k|k-1}")
                st.divider()
                st.markdown("### Métricas")
                st.latex(r"\mathrm{MSE} = \frac{1}{N}\sum_{k=1}^{N}(x_k - \hat{x}_k)^2")
                st.latex(r"\mathrm{SNR} = 10\log_{10}\!\left(\frac{\|x\|^2}{\|x-\hat{x}\|^2}\right)\;\mathrm{[dB]}")
                st.divider()
                st.caption("Q pequeño → señal varía lentamente (más suavizado).")
                st.caption("R pequeño → alta confianza en la medición (menos filtrado).")

        main_col, side_col = st.columns([2.5, 1.0], gap="large")
        with side_col:
            _formulas_kalman()

        with main_col:
            with st.container(border=True):
                st.subheader("🎙️ Reconstrucción de Señal de Voz — Filtro de Kalman")
                st.caption("Compara el filtro de Kalman contra interpolación polinómica y spline cúbico.")

            with st.container(border=True):
                st.markdown("### ⚙️ Parámetros de simulación")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown("**Señal y ruido**")
                    n_muestras = st.slider("Muestras de señal", 100, 1000, 400, 50, key="kal_n")
                    snr_entrada = st.slider("SNR entrada (dB)", -5, 30, 10, 1, key="kal_snr")
                    freq_fund = st.slider("Frecuencia fundamental (Hz)", 80, 300, 150, 10, key="kal_freq")
                    seed = st.number_input("Semilla aleatoria", value=42, min_value=0, max_value=9999, step=1, key="kal_seed")
                with col_p2:
                    st.markdown("**Parámetros del Filtro de Kalman**")
                    Q_exp = st.slider("Q (ruido proceso) — exponente 10^x", -8, 0, -2, 1, key="kal_Q")
                    R_exp = st.slider("R (ruido medición) — exponente 10^x", -3, 1, -2, 1, key="kal_R")
                    Q_val = 10.0 ** Q_exp; R_val = 10.0 ** R_exp
                    st.info(f"Q = {Q_val:.2e}   |   R = {R_val:.2e}")
                    st.markdown("**Parámetros de métodos clásicos**")
                    grado_poly = st.slider("Grado polinomio", 2, 20, 8, 1, key="kal_grado")
                    factor_spline = st.slider("Factor submuestreo spline", 2, 20, 5, 1, key="kal_spline")

            run_kal = st.button("▶ Ejecutar comparación", type="primary", key="kal_run")

            if run_kal:
                try:
                    t_arr, x_original = generar_senal_voz(n_muestras=n_muestras, fs=8000.0, freq_fundamental=float(freq_fund), seed=int(seed))
                    z_ruidosa = agregar_ruido(x_original, snr_db=float(snr_entrada), seed=int(seed))
                    res = comparar_metodos(t_arr, x_original, z_ruidosa, grado_poly=grado_poly, factor_spline=factor_spline, Q=Q_val, R=R_val)
                    metricas = res["metricas"]

                    iconos = {"Señal ruidosa": "🔴", "Polinomio": "🟠", "Spline cúbico": "🟡", "Kalman": "🟢"}
                    colores = {"Señal ruidosa": "#ef4444", "Polinomio": "#f97316", "Spline cúbico": "#eab308", "Kalman": "#22c55e"}

                    with st.container(border=True):
                        st.markdown("### 📊 Métricas de reconstrucción")
                        mc = st.columns(len(metricas))
                        for col_m, (nombre, vals) in zip(mc, metricas.items()):
                            snr_v = vals["snr"]
                            snr_str = f"{snr_v:.2f} dB" if np.isfinite(snr_v) else "∞"
                            col_m.metric(f"{iconos[nombre]} {nombre}", f"SNR: {snr_str}", f"MSE: {vals['mse']:.5f}", delta_color="inverse")
                        tabla_rows = []
                        for nombre, vals in metricas.items():
                            snr_v = vals["snr"]
                            tabla_rows.append({"Método": f"{iconos[nombre]} {nombre}", "MSE": f"{vals['mse']:.6f}",
                                               "SNR (dB)": f"{snr_v:.4f}" if np.isfinite(snr_v) else "∞"})
                        st.dataframe(pd.DataFrame(tabla_rows), hide_index=True, use_container_width=True)

                    t_ms = t_arr * 1000
                    def _serie(t_ms, y, nombre, color):
                        return pd.DataFrame({"t_ms": t_ms, "amplitud": y, "Señal": nombre, "_color": color})

                    with st.container(border=True):
                        st.markdown("### 📈 Comparación de señales")
                        df_all = pd.concat([
                            _serie(t_ms, x_original,        "Original",       "#6366f1"),
                            _serie(t_ms, z_ruidosa,          "Ruidosa",        "#ef4444"),
                            _serie(t_ms, res["kalman"],      "Kalman",         "#22c55e"),
                            _serie(t_ms, res["spline"],      "Spline cúbico",  "#eab308"),
                            _serie(t_ms, res["polinomio"],   "Polinomio",      "#f97316"),
                        ], ignore_index=True)
                        color_scale = alt.Scale(
                            domain=["Original", "Ruidosa", "Kalman", "Spline cúbico", "Polinomio"],
                            range=["#6366f1", "#ef4444", "#22c55e", "#eab308", "#f97316"])
                        chart_all = (alt.Chart(df_all).mark_line(strokeWidth=1.8, opacity=0.9).encode(
                            x=alt.X("t_ms:Q", title="Tiempo (ms)"), y=alt.Y("amplitud:Q"),
                            color=alt.Color("Señal:N", scale=color_scale),
                            tooltip=[alt.Tooltip("t_ms:Q", title="t (ms)", format=".2f"), alt.Tooltip("amplitud:Q", format=".5f"), "Señal:N"],
                        ).properties(height=380, title="Todas las señales").interactive())
                        st.altair_chart(chart_all, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 🔬 Detalle por método")
                        tab1, tab2, tab3 = st.tabs(["Polinomio", "Spline cúbico", "Kalman"])

                        def _chart_detalle(y_metodo, nombre_metodo, color_metodo):
                            df_det = pd.concat([
                                _serie(t_ms, x_original,  "Original",     "#6366f1"),
                                _serie(t_ms, z_ruidosa,   "Ruidosa",      "#ef4444"),
                                _serie(t_ms, y_metodo,    nombre_metodo,  color_metodo),
                            ], ignore_index=True)
                            return (alt.Chart(df_det).mark_line(strokeWidth=1.8, opacity=0.85).encode(
                                x=alt.X("t_ms:Q", title="Tiempo (ms)"), y=alt.Y("amplitud:Q"),
                                color=alt.Color("Señal:N", scale=alt.Scale(
                                    domain=["Original", "Ruidosa", nombre_metodo],
                                    range=["#6366f1", "#ef4444", color_metodo])),
                                tooltip=[alt.Tooltip("t_ms:Q", format=".2f"), alt.Tooltip("amplitud:Q", format=".5f"), "Señal:N"],
                            ).properties(height=320, title=f"Original vs Ruidosa vs {nombre_metodo}").interactive())

                        with tab1:
                            st.altair_chart(_chart_detalle(res["polinomio"], "Polinomio", "#f97316"), use_container_width=True)
                            st.info(f"Polinomio grado {grado_poly}  →  MSE = {metricas['Polinomio']['mse']:.6f}  |  SNR = {metricas['Polinomio']['snr']:.2f} dB")
                        with tab2:
                            st.altair_chart(_chart_detalle(res["spline"], "Spline cúbico", "#eab308"), use_container_width=True)
                            st.info(f"Spline (1 nodo cada {factor_spline} muestras)  →  MSE = {metricas['Spline cúbico']['mse']:.6f}  |  SNR = {metricas['Spline cúbico']['snr']:.2f} dB")
                        with tab3:
                            st.altair_chart(_chart_detalle(res["kalman"], "Kalman", "#22c55e"), use_container_width=True)
                            st.info(f"Filtro de Kalman (Q={Q_val:.2e}, R={R_val:.2e})  →  MSE = {metricas['Kalman']['mse']:.6f}  |  SNR = {metricas['Kalman']['snr']:.2f} dB")

                    with st.container(border=True):
                        st.markdown("### 📉 Error residual |x_original − x̂|")
                        df_err = pd.concat([
                            _serie(t_ms, np.abs(x_original - z_ruidosa),    "Ruidosa",   "#ef4444"),
                            _serie(t_ms, np.abs(x_original - res["polinomio"]), "Polinomio", "#f97316"),
                            _serie(t_ms, np.abs(x_original - res["spline"]),    "Spline",    "#eab308"),
                            _serie(t_ms, np.abs(x_original - res["kalman"]),    "Kalman",    "#22c55e"),
                        ], ignore_index=True)
                        chart_err = (alt.Chart(df_err).mark_line(strokeWidth=1.5, opacity=0.85).encode(
                            x=alt.X("t_ms:Q", title="Tiempo (ms)"), y=alt.Y("amplitud:Q", title="|error|"),
                            color=alt.Color("Señal:N", scale=alt.Scale(
                                domain=["Ruidosa", "Polinomio", "Spline", "Kalman"],
                                range=["#ef4444", "#f97316", "#eab308", "#22c55e"])),
                            tooltip=[alt.Tooltip("t_ms:Q", format=".2f"), alt.Tooltip("amplitud:Q", title="|error|", format=".6f"), "Señal:N"],
                        ).properties(height=320, title="Error absoluto por método").interactive())
                        st.altair_chart(chart_err, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 🏆 Comparación SNR (barras)")
                        snr_vals = {n: v["snr"] for n, v in metricas.items() if np.isfinite(v["snr"])}
                        df_snr = pd.DataFrame([{"Método": k, "SNR (dB)": v} for k, v in snr_vals.items()])
                        color_map = {"Señal ruidosa": "#ef4444", "Polinomio": "#f97316", "Spline cúbico": "#eab308", "Kalman": "#22c55e"}
                        df_snr["color"] = df_snr["Método"].map(color_map)
                        chart_snr = (alt.Chart(df_snr).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                            x=alt.X("Método:N", sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("SNR (dB):Q"),
                            color=alt.Color("Método:N", scale=alt.Scale(
                                domain=list(color_map.keys()), range=list(color_map.values())), legend=None),
                            tooltip=["Método:N", alt.Tooltip("SNR (dB):Q", format=".3f")],
                        ).properties(height=300, title="SNR de salida por método (mayor es mejor)"))
                        st.altair_chart(chart_snr, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 💡 Interpretación de resultados")
                        mejor = max(metricas.items(), key=lambda kv: kv[1]["snr"])
                        peor  = min(metricas.items(), key=lambda kv: kv[1]["snr"])
                        st.markdown(f"- **Mejor método:** {iconos[mejor[0]]} **{mejor[0]}** con SNR = {mejor[1]['snr']:.2f} dB y MSE = {mejor[1]['mse']:.6f}")
                        st.markdown(f"- **Peor método:** {iconos[peor[0]]} **{peor[0]}** con SNR = {peor[1]['snr']:.2f} dB y MSE = {peor[1]['mse']:.6f}")
                        st.markdown("> *Mientras los métodos de interpolación intentan describir los datos observados, el filtro de Kalman busca **inferir la realidad que los genera.***")

                except Exception as e:
                    import traceback
                    st.error(f"Error en la simulación: {e}")
                    st.code(traceback.format_exc())
