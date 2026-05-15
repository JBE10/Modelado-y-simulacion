"""monte_carlo_module.py — Monte Carlo UI Module."""
import math
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, make_fn_2d, _formulas_panel, parse_number_cell
from montecarlo import (
    estimar_pi, integracion_1d_mc, integracion_2d_mc, multi_run_1d, simular_monty_hall,
)


class MonteCarloModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Monte Carlo"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            modo = st.radio("Modo de Simulación", [
                "Estimación de π", "Integración 1D", "Integración 2D",
                "Convergencia Progresiva", "Histograma CLT", "Comparar 2 Integrales", "Monty Hall (Juego)"
            ], horizontal=True)

            if modo == "Estimación de π":
                self._pi()
            elif modo == "Integración 1D":
                self._integracion_1d()
            elif modo == "Integración 2D":
                self._integracion_2d()
            elif modo == "Convergencia Progresiva":
                self._convergencia()
            elif modo == "Histograma CLT":
                self._clt()
            elif modo == "Comparar 2 Integrales":
                self._comparar()
            elif modo == "Monty Hall (Juego)":
                self._monty_hall()

        with side_col:
            _formulas_panel("Monte Carlo")

    # ── Sub-modes ──────────────────────────────────────────────────────────────
    def _pi(self):
        with st.container(border=True):
            st.subheader("Monte Carlo: Aciertos en un Círculo")
            with st.form("mc_pi"):
                c1, c2 = st.columns(2)
                with c1:
                    n_puntos = st.number_input("Número de Puntos (N)", value=100000, step=10000)
                    animar_pi = st.checkbox("Animar caída de puntos", value=False)
                with c2:
                    semilla = st.number_input("Semilla Aleatoria", value=42, min_value=0, step=1)
                    usar_semilla = st.checkbox("Fijar semilla para reproducibilidad", value=True)
                run_pi = st.form_submit_button("Simular", type="primary")

        if run_pi:
            seed = int(semilla) if usar_semilla else None
            res_pi = estimar_pi(int(n_puntos), seed)
            with st.container(border=True):
                cols = st.columns(3)
                cols[0].metric("Puntos Disparados (N)", f"{res_pi['num_puntos']}")
                cols[1].metric("Puntos dentro (Acertados)", f"{res_pi['puntos_dentro']}")
                cols[2].metric("Aproximación de π", f"{res_pi['pi_estimado']:.8f}")
            with st.container(border=True):
                st.markdown("#### Resultado Detallado")
                st.latex(rf"\pi \approx 4 \times \frac{{{res_pi['puntos_dentro']}}}{{{res_pi['num_puntos']}}} = {res_pi['pi_estimado']:.10f}")
                st.latex(rf"\left|\,\pi_{{\text{{estimado}}}} - \pi_{{\text{{real}}}}\,\right| = {res_pi['error_vs_pi']:.10f}")
                st.latex(rf"\text{{Error Estándar}} = {res_pi['error_estandar']:.10f}")
                st.latex(rf"\text{{IC}}^{{95\%}} = \left[{res_pi['pi_estimado'] - 1.96*res_pi['error_estandar']:.10f},\ {res_pi['pi_estimado'] + 1.96*res_pi['error_estandar']:.10f}\right]")
            pts = res_pi.get("puntos_grafica", [])
            if pts:
                df = pd.DataFrame(pts)
                st.caption(f"Mostrando los primeros {len(pts)} puntos generados:")
                chart = alt.Chart(df).mark_circle(size=15).encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[-1, 1])),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[-1, 1])),
                    color=alt.Color("estado:N", scale=alt.Scale(domain=["Dentro", "Fuera"], range=["#22c55e", "#ef4444"]))
                ).properties(height=400, width=400).interactive()
                st.altair_chart(chart, use_container_width=True)

    def _integracion_1d(self):
        with st.container(border=True):
            st.subheader("Monte Carlo: Integración 1D")
            with st.form("mc_1d"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f = st.text_input("f(x)", "exp(-x**2)", key="mc_f1")
                    a_str = st.text_input("Límite inferior a", "0")
                    b_str = st.text_input("Límite superior b", "2")
                with c2:
                    n_puntos = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000)
                    confianza = st.selectbox("Intervalo de Confianza", ["90%", "95%", "99%"], index=1)
                    semilla = st.number_input("Semilla Aleatoria", value=42, min_value=0, step=1)
                    usar_semilla = st.checkbox("Fijar semilla", value=True)
                run_1d = st.form_submit_button("Integrar", type="primary")

        if run_1d:
            try:
                f_fn = make_fn(expr_f)
                a = parse_number_cell(a_str)
                b = parse_number_cell(b_str)
                seed = int(semilla) if usar_semilla else None
                res_1d = integracion_1d_mc(f_fn, a, b, int(n_puntos), confianza, seed)
                with st.container(border=True):
                    st.success(f"Cálculo finalizado tras {n_puntos} muestras.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Integral Estimada Î", f"{res_1d['integral']:.8f}")
                    m2.metric("Margen de Error (±)", f"{res_1d['margen_error']:.8f}")
                    m3.metric(f"IC al {confianza}", "Ver Tabla Abajo")
                    st.latex(rf"\hat{{I}} = (b-a) \cdot \bar{{f}} = {res_1d['integral']:.10f}")
                    st.latex(rf"\sigma = {res_1d['sigma']:.10f}")
                    conf_latex = confianza.replace("%", r"\%")
                    st.latex(rf"\text{{IC}}^{{{conf_latex}}} = \left[ {res_1d['ic_inferior']:.10f},\ {res_1d['ic_superior']:.10f} \right]")
                graf = res_1d.get("puntos_grafica", [])
                if graf:
                    df = pd.DataFrame(graf)
                    chart = alt.Chart(df).mark_circle(opacity=0.3, size=20, color="#3b82f6").encode(
                        x=alt.X("x:Q", scale=alt.Scale(domain=[a, b])), y=alt.Y("y:Q", title="f(x)")
                    ).properties(height=350).interactive()
                    st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    def _integracion_2d(self):
        with st.container(border=True):
            st.subheader("Monte Carlo: Integración 2D Multivariable")
            with st.form("mc_2d"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f2 = st.text_input("f(x, y)", "sin(x) * cos(y)")
                    ax_str = st.text_input("Límite x inferior (a)", "0")
                    bx_str = st.text_input("Límite x superior (b)", "pi/2")
                with c2:
                    cy_str = st.text_input("Límite y inferior (c)", "0")
                    dy_str = st.text_input("Límite y superior (d)", "pi/2")
                    n_puntos = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000)
                    confianza = st.selectbox("Intervalo de Confianza", ["90%", "95%", "99%"], index=2)
                    semilla = st.number_input("Semilla", value=42, min_value=0, step=1)
                    usar_semilla = st.checkbox("Fijar semilla", value=True)
                run_2d = st.form_submit_button("Calcular Volumen/Integral", type="primary")

        if run_2d:
            try:
                f_2d_fn = make_fn_2d(expr_f2)
                a = parse_number_cell(ax_str); b = parse_number_cell(bx_str)
                c = parse_number_cell(cy_str); d = parse_number_cell(dy_str)
                seed = int(semilla) if usar_semilla else None
                res_2d = integracion_2d_mc(f_2d_fn, a, b, c, d, int(n_puntos), confianza, seed)
                with st.container(border=True):
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Integral Estimada Î", f"{res_2d['integral']:.8f}")
                    m2.metric("Margen de Error (±)", f"{res_2d['margen_error']:.8f}")
                    m3.metric(f"IC al {confianza}", "Ver Abajo")
                    conf_latex = confianza.replace("%", r"\%")
                    st.latex(rf"\text{{IC}}^{{{conf_latex}}} = \left[ {res_2d['ic_inferior']:.10f},\ {res_2d['ic_superior']:.10f} \right]")
            except Exception as e:
                st.error(f"Error: {e}")

    def _convergencia(self):
        with st.container(border=True):
            st.subheader("Convergencia Progresiva de Monte Carlo")
            with st.form("mc_conv"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f_conv = st.text_input("f(x)", "x**2", key="mc_conv_f")
                    a_conv = st.text_input("a", "0", key="mc_conv_a")
                    b_conv = st.text_input("b", "2", key="mc_conv_b")
                with c2:
                    n_conv = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000, key="mc_conv_n")
                    conf_conv = st.selectbox("IC", ["90%", "95%", "99%"], index=1, key="mc_conv_ic")
                    semilla_conv = st.number_input("Semilla", value=42, min_value=0, step=1, key="mc_conv_s")
                    usar_semilla_conv = st.checkbox("Fijar semilla", value=True, key="mc_conv_fix")
                run_conv = st.form_submit_button("Graficar Convergencia", type="primary")

        if run_conv:
            try:
                f_fn = make_fn(expr_f_conv)
                a = parse_number_cell(a_conv)
                b = parse_number_cell(b_conv)
                seed = int(semilla_conv) if usar_semilla_conv else None
                res_conv = integracion_1d_mc(f_fn, a, b, int(n_conv), conf_conv, seed)
                snaps = res_conv.get("snapshots", [])
                if snaps:
                    with st.container(border=True):
                        st.markdown("### Evolución de la Estimación")
                        df_snap = pd.DataFrame(snaps)
                        linea = alt.Chart(df_snap).mark_line(color="#2563eb", strokeWidth=2).encode(
                            x=alt.X("n:Q", title="N muestras"), y=alt.Y("integral:Q", title="Integral Estimada"))
                        banda = alt.Chart(df_snap).mark_area(opacity=0.2, color="#3b82f6").encode(
                            x="n:Q", y="ic_inf:Q", y2="ic_sup:Q")
                        valor_final = alt.Chart(pd.DataFrame([{"y": res_conv["integral"]}])).mark_rule(
                            color="#dc2626", strokeDash=[5, 3], strokeWidth=2).encode(y="y:Q")
                        st.altair_chart((banda + linea + valor_final).properties(height=400).interactive(), use_container_width=True)
                        st.latex(rf"\hat{{I}}_{{\text{{final}}}} = {res_conv['integral']:.10f}")
            except Exception as e:
                st.error(f"Error: {e}")

    def _clt(self):
        with st.container(border=True):
            st.subheader("Histograma: Teorema Central del Límite")
            with st.form("mc_clt"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f_clt = st.text_input("f(x)", "x**2", key="mc_clt_f")
                    a_clt = st.text_input("a", "0", key="mc_clt_a")
                    b_clt = st.text_input("b", "2", key="mc_clt_b")
                with c2:
                    n_clt = st.number_input("N por corrida", value=5000, step=1000, key="mc_clt_n")
                    k_clt = st.number_input("Número de corridas (K)", value=200, step=50, key="mc_clt_k")
                run_clt = st.form_submit_button("Simular K corridas", type="primary")

        if run_clt:
            try:
                f_fn = make_fn(expr_f_clt)
                a = parse_number_cell(a_clt); b = parse_number_cell(b_clt)
                with st.spinner(f"Ejecutando {k_clt} simulaciones MC..."):
                    resultados = multi_run_1d(f_fn, a, b, int(n_clt), int(k_clt))
                with st.container(border=True):
                    media = np.mean(resultados); desv = np.std(resultados, ddof=1)
                    st.latex(rf"\bar{{I}} = {media:.10f}")
                    st.latex(rf"s = {desv:.10f}")
                    df_hist = pd.DataFrame({"Integral Estimada": resultados})
                    hist_chart = alt.Chart(df_hist).mark_bar(opacity=0.7, color="#6366f1").encode(
                        alt.X("Integral Estimada:Q", bin=alt.Bin(maxbins=40), title="Valor de la Integral"),
                        y=alt.Y("count()", title="Frecuencia")
                    ).properties(height=350, title=f"Histograma de {int(k_clt)} corridas MC (N={int(n_clt)} c/u)")
                    media_rule = alt.Chart(pd.DataFrame([{"x": media}])).mark_rule(
                        color="#dc2626", strokeWidth=2, strokeDash=[5, 3]).encode(x="x:Q")
                    st.altair_chart(hist_chart + media_rule, use_container_width=True)
                    st.caption("La distribución se aproxima a una Normal (TCL). Línea roja: media.")
            except Exception as e:
                st.error(f"Error: {e}")

    def _comparar(self):
        with st.container(border=True):
            st.subheader("Comparar dos Integrales con Monte Carlo")
            with st.form("mc_cmp2"):
                st.markdown("##### Integral A")
                ca1, ca2, ca3 = st.columns(3)
                with ca1: expr_a = st.text_input("f(x) — Integral A", "x**2", key="mc_a_f")
                with ca2: a_a = st.text_input("a (A)", "0", key="mc_a_a"); b_a = st.text_input("b (A)", "2", key="mc_a_b")
                with ca3: label_a = st.text_input("Etiqueta A", "Integral A", key="mc_a_label")
                st.markdown("##### Integral B")
                cb1, cb2, cb3 = st.columns(3)
                with cb1: expr_b = st.text_input("f(x) — Integral B", "sin(x)", key="mc_b_f")
                with cb2: a_b = st.text_input("a (B)", "0", key="mc_b_a"); b_b = st.text_input("b (B)", "pi", key="mc_b_b")
                with cb3: label_b = st.text_input("Etiqueta B", "Integral B", key="mc_b_label")
                st.markdown("##### Parámetros comunes")
                cp1, cp2, cp3 = st.columns(3)
                with cp1: n_cmp2 = st.number_input("N evaluaciones", value=100000, step=10000, key="mc_cmp2_n")
                with cp2: conf_cmp2 = st.selectbox("IC", ["90%", "95%", "99%"], index=1, key="mc_cmp2_ic")
                with cp3:
                    semilla_cmp2 = st.number_input("Semilla", value=42, min_value=0, step=1, key="mc_cmp2_s")
                    usar_semilla_cmp2 = st.checkbox("Fijar semilla", value=True, key="mc_cmp2_fix")
                run_cmp2 = st.form_submit_button("Comparar Integrales", type="primary")

        if run_cmp2:
            try:
                seed = int(semilla_cmp2) if usar_semilla_cmp2 else None
                fa = make_fn(expr_a); fb = make_fn(expr_b)
                aa = parse_number_cell(a_a); ba = parse_number_cell(b_a)
                ab = parse_number_cell(a_b); bb = parse_number_cell(b_b)
                res_a = integracion_1d_mc(fa, aa, ba, int(n_cmp2), conf_cmp2, seed)
                res_b = integracion_1d_mc(fb, ab, bb, int(n_cmp2), conf_cmp2, seed)
                with st.container(border=True):
                    st.markdown("### Tabla Comparativa")
                    conf_l = conf_cmp2.replace("%", r"\%")
                    st.table(pd.DataFrame([
                        {"Integral": label_a, "f(x)": expr_a, "[a,b]": f"[{aa},{ba}]",
                         "Î": f"{res_a['integral']:.10f}", "Margen ±": f"{res_a['margen_error']:.10f}",
                         f"IC inf": f"{res_a['ic_inferior']:.10f}", "IC sup": f"{res_a['ic_superior']:.10f}"},
                        {"Integral": label_b, "f(x)": expr_b, "[a,b]": f"[{ab},{bb}]",
                         "Î": f"{res_b['integral']:.10f}", "Margen ±": f"{res_b['margen_error']:.10f}",
                         f"IC inf": f"{res_b['ic_inferior']:.10f}", "IC sup": f"{res_b['ic_superior']:.10f}"},
                    ]))
                snaps_a = res_a.get("snapshots", []); snaps_b = res_b.get("snapshots", [])
                if snaps_a and snaps_b:
                    with st.container(border=True):
                        st.markdown("### Convergencia Comparada")
                        df_a = pd.DataFrame(snaps_a); df_a["serie"] = label_a
                        df_b = pd.DataFrame(snaps_b); df_b["serie"] = label_b
                        df_conv = pd.concat([df_a, df_b], ignore_index=True)
                        chart_conv = alt.Chart(df_conv).mark_line(strokeWidth=2).encode(
                            x="n:Q", y="integral:Q", color="serie:N", tooltip=["n", "serie", "integral"]
                        ).properties(height=380).interactive()
                        st.altair_chart(chart_conv, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    def _monty_hall(self):
        with st.container(border=True):
            st.subheader("🐐 El Casino de Monty Hall (Destructor de Intuición)")
            with st.form("mc_monty_hall"):
                c1, c2 = st.columns(2)
                with c1:
                    num_partidas = st.number_input("Número de Partidas", min_value=100, value=10000, step=1000, key="mh_partidas")
                with c2:
                    semilla_mh = st.number_input("Semilla", value=42, min_value=0, step=1, key="mh_seed")
                    usar_semilla_mh = st.checkbox("Fijar semilla", value=True, key="mh_usar_seed")
                    animar_mh = st.checkbox("Animar simulación", value=True, key="mh_animar")
                run_mh = st.form_submit_button("Simular las partidas", type="primary")

        if run_mh:
            try:
                seed = int(semilla_mh) if usar_semilla_mh else None
                res_mh = simular_monty_hall(int(num_partidas), semilla=seed)
                with st.container(border=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Partidas Jugadas", f"{res_mh['num_partidas']:,}")
                    col2.metric("Estrategia: MANTENER", f"{res_mh['tasa_mantener']*100:.2f}%", f"{res_mh['wins_mantener']} ganadas", delta_color="off")
                    col3.metric("Estrategia: CAMBIAR", f"{res_mh['tasa_cambiar']*100:.2f}%", f"{res_mh['wins_cambiar']} ganadas")
                historial = res_mh.get("historial", [])
                if historial:
                    with st.container(border=True):
                        st.markdown("### La Carrera: Cambiar vs Mantener")
                        df_h = pd.DataFrame(historial).melt("partida", var_name="Estrategia", value_name="Win Rate")
                        df_h["Estrategia"] = df_h["Estrategia"].replace({
                            "win_rate_mantener": "Mantener mi puerta original",
                            "win_rate_cambiar": "Cambiar de puerta"
                        })
                        chart = alt.Chart(df_h).mark_line(strokeWidth=3).encode(
                            x=alt.X("partida:Q", title="Nº de Partida"),
                            y=alt.Y("Win Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                            color=alt.Color("Estrategia:N", scale=alt.Scale(
                                domain=["Mantener mi puerta original", "Cambiar de puerta"],
                                range=["#ef4444", "#22c55e"])),
                            tooltip=[alt.Tooltip("partida:Q"), alt.Tooltip("Estrategia:N"), alt.Tooltip("Win Rate:Q", format=".2%")]
                        ).properties(height=400).interactive()
                        st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
