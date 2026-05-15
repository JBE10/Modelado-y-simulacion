"""simulacion_sir_module.py — SIR Epidemic Simulation UI Module."""
import time as _time
import numpy as _np
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import _formulas_panel
from sir_model import sir_euler, sir_rk4, calcular_metricas


class SirModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Simulación SIR (Epidemia)"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("🦠 Simulación Epidemiológica — Modelo SIR")
                st.caption("Simula la propagación de una enfermedad usando el sistema SIR, resuelto con **RK4**.")
                with st.form("sir_form"):
                    st.markdown("#### Parámetros de la epidemia")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        N_pop = st.number_input("Población total (N)", value=10000, min_value=100, step=100, key="sir_N")
                        I0_sir = st.number_input("Infectados iniciales (I₀)", value=10, min_value=1, step=1, key="sir_I0")
                        R0_init_sir = st.number_input("Recuperados iniciales", value=0, min_value=0, step=1, key="sir_R0i")
                    with c2:
                        beta_sir = st.number_input("β (tasa de contagio)", value=0.3, min_value=0.01, max_value=5.0, step=0.01, format="%.3f", key="sir_beta")
                        gamma_sir = st.number_input("γ (tasa de recuperación)", value=0.1, min_value=0.01, max_value=5.0, step=0.01, format="%.3f", key="sir_gamma")
                    with c3:
                        t_max_sir = st.number_input("Días a simular", value=160, min_value=10, step=10, key="sir_tmax")
                        h_sir = st.number_input("Paso (h) en días", value=0.5, min_value=0.01, max_value=5.0, step=0.1, format="%.2f", key="sir_h")
                        comparar_metodos_sir = st.checkbox("Comparar Euler vs RK4", value=False, key="sir_cmp")
                    run_sir = st.form_submit_button("Simular Epidemia", type="primary")

            if run_sir:
                try:
                    res_rk4 = sir_rk4(beta_sir, gamma_sir, N_pop, I0_sir, R0_init_sir, t_max_sir, h_sir)
                    metricas = calcular_metricas(res_rk4, beta_sir, gamma_sir, N_pop)

                    with st.container(border=True):
                        st.markdown("### 📊 Métricas Clave")
                        m1, m2, m3, m4 = st.columns(4)
                        r0_val = metricas["R0"]
                        m1.metric("R₀", f"{r0_val:.2f}", "Epidemia" if r0_val > 1 else "Se extingue")
                        m2.metric("Pico de Infectados", f"{metricas['pico_infectados']:.0f}", f"{metricas['pico_pct']:.1f}% de N")
                        m3.metric("Día del Pico", f"Día {metricas['dia_pico']:.0f}")
                        m4.metric("Total Infectados", f"{metricas['total_infectados_pct']:.1f}%", f"Inmunidad rebaño: {metricas['umbral_rebano_pct']:.1f}%")

                    with st.container(border=True):
                        st.markdown("### 📐 Sistema de Ecuaciones Diferenciales (tu PVI)")
                        st.caption("Problema de Valor Inicial resuelto con **Runge-Kutta de orden 4**:")
                        st.latex(rf"\frac{{dS}}{{dt}} = -\frac{{{beta_sir}}} {{{N_pop}}} \cdot S(t) \cdot I(t)\quad,\quad S(0) = {N_pop - I0_sir - R0_init_sir}")
                        st.latex(rf"\frac{{dI}}{{dt}} = \frac{{{beta_sir}}}{{{N_pop}}} \cdot S(t) \cdot I(t) - {gamma_sir} \cdot I(t)\quad,\quad I(0) = {I0_sir}")
                        st.latex(rf"\frac{{dR}}{{dt}} = {gamma_sir} \cdot I(t)\quad,\quad R(0) = {R0_init_sir}")
                        st.latex(rf"\beta = {beta_sir},\quad \gamma = {gamma_sir},\quad R_0 = \frac{{\beta}}{{\gamma}} = {metricas['R0']:.2f},\quad h = {h_sir},\quad t \in [0,\,{t_max_sir}]")

                    with st.container(border=True):
                        st.markdown("### 📋 Tabla de Iteraciones (RK4)")
                        st.caption("Valores de S(t), I(t), R(t) calculados paso a paso con RK4.")
                        df_rk4_table = pd.DataFrame({
                            "Paso": list(range(len(res_rk4["t"]))),
                            "t (días)": [f"{v:.2f}" for v in res_rk4["t"]],
                            "S(t)": [f"{v:.2f}" for v in res_rk4["S"]],
                            "I(t)": [f"{v:.2f}" for v in res_rk4["I"]],
                            "R(t)": [f"{v:.2f}" for v in res_rk4["R"]],
                            "S+I+R": [f"{s+i+r:.0f}" for s, i, r in zip(res_rk4["S"], res_rk4["I"], res_rk4["R"])],
                        })
                        _step_show = max(1, len(df_rk4_table) // 50)
                        st.dataframe(df_rk4_table.iloc[::_step_show], hide_index=True, use_container_width=True, height=400)
                        st.caption(f"Total de pasos RK4: **{len(res_rk4['t'])-1}** (mostrando cada {_step_show}). h = {h_sir} días.")

                    df_sir = pd.DataFrame({"Día": res_rk4["t"], "Susceptibles": res_rk4["S"], "Infectados": res_rk4["I"], "Recuperados": res_rk4["R"]})

                    with st.container(border=True):
                        st.markdown("### 📈 Evolución de la Epidemia (RK4)")
                        df_melt = df_sir.melt("Día", var_name="Grupo", value_name="Personas")
                        color_scale = alt.Scale(domain=["Susceptibles", "Infectados", "Recuperados"], range=["#3b82f6", "#ef4444", "#22c55e"])
                        chart_sir = alt.Chart(df_melt).mark_line(strokeWidth=2.5).encode(
                            x=alt.X("Día:Q", title="Tiempo (días)"), y=alt.Y("Personas:Q"),
                            color=alt.Color("Grupo:N", scale=color_scale),
                            tooltip=["Día", "Grupo", alt.Tooltip("Personas:Q", format=",.0f")]
                        ).properties(height=420).interactive()
                        df_pico = pd.DataFrame({"x": [metricas["dia_pico"]]})
                        rule_pico = alt.Chart(df_pico).mark_rule(color="#ef4444", strokeDash=[6, 3], strokeWidth=1.5).encode(x="x:Q")
                        label_pico = alt.Chart(df_pico).mark_text(align="left", dx=5, dy=-10, fontSize=11, color="#ef4444", fontWeight="bold").encode(
                            x="x:Q", text=alt.value(f"Pico: día {metricas['dia_pico']:.0f}"))
                        st.altair_chart(chart_sir + rule_pico + label_pico, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 🔴 Curva de Infectados (detalle)")
                        df_inf = pd.DataFrame({"Día": res_rk4["t"], "Infectados": res_rk4["I"]})
                        chart_inf = alt.Chart(df_inf).mark_area(
                            color=alt.Gradient(gradient="linear",
                                stops=[alt.GradientStop(color="#fecaca", offset=0), alt.GradientStop(color="#ef4444", offset=1)],
                                x1=1, x2=1, y1=1, y2=0),
                            line={"color": "#dc2626", "strokeWidth": 2},
                        ).encode(
                            x=alt.X("Día:Q", title="Tiempo (días)"),
                            y=alt.Y("Infectados:Q", title="Personas infectadas"),
                            tooltip=["Día", alt.Tooltip("Infectados:Q", format=",.0f")],
                        ).properties(height=300).interactive()
                        st.altair_chart(chart_inf, use_container_width=True)

                    with st.container(border=True):
                        st.markdown("### 📋 Tabla de Valores")
                        step_display = max(1, len(df_sir) // 40)
                        st.dataframe(df_sir.iloc[::step_display].style.format(
                            {"Día": "{:.1f}", "Susceptibles": "{:,.0f}", "Infectados": "{:,.0f}", "Recuperados": "{:,.0f}"}
                        ), hide_index=True, use_container_width=True)

                    if comparar_metodos_sir:
                        res_euler = sir_euler(beta_sir, gamma_sir, N_pop, I0_sir, R0_init_sir, t_max_sir, h_sir)
                        with st.container(border=True):
                            st.markdown("### ⚖️ Comparación: Euler vs RK4 (Infectados)")
                            df_cmp = pd.concat([
                                pd.DataFrame({"Día": res_euler["t"], "Infectados": res_euler["I"], "Método": "Euler"}),
                                pd.DataFrame({"Día": res_rk4["t"], "Infectados": res_rk4["I"], "Método": "RK4"}),
                            ], ignore_index=True)
                            chart_cmp = alt.Chart(df_cmp).mark_line(strokeWidth=2).encode(
                                x=alt.X("Día:Q", title="Tiempo (días)"), y=alt.Y("Infectados:Q"),
                                color=alt.Color("Método:N", scale=alt.Scale(domain=["Euler", "RK4"], range=["#f59e0b", "#7c3aed"])),
                                strokeDash=alt.StrokeDash("Método:N", scale=alt.Scale(domain=["Euler", "RK4"], range=[[6, 3], [0]])),
                                tooltip=["Día", "Método", alt.Tooltip("Infectados:Q", format=",.0f")],
                            ).properties(height=350).interactive()
                            st.altair_chart(chart_cmp, use_container_width=True)
                            met_euler = calcular_metricas(res_euler, beta_sir, gamma_sir, N_pop)
                            diff_pico = abs(metricas["pico_infectados"] - met_euler["pico_infectados"])
                            diff_dia = abs(metricas["dia_pico"] - met_euler["dia_pico"])
                            st.caption(f"Diferencia en pico: **{diff_pico:,.0f} personas** | Diferencia en día del pico: **{diff_dia:.1f} días** | Con h = {h_sir}")
                            if diff_pico > 10:
                                st.info("💡 Euler acumula error numérico. Probá aumentar h (e.g. h=2) para ver cómo RK4 mantiene precisión.")

                    with st.container(border=True):
                        st.markdown("### 🧠 Interpretación")
                        r0_val = metricas["R0"]
                        if r0_val > 1:
                            st.warning(
                                f"Con **R₀ = {r0_val:.2f}**, cada infectado contagia en promedio a **{r0_val:.1f} personas**.\n\n"
                                f"El **pico** ocurre en el **día {metricas['dia_pico']:.0f}** con **{metricas['pico_infectados']:,.0f} infectados simultáneos** ({metricas['pico_pct']:.1f}% de la población).\n\n"
                                f"Al final, el **{metricas['total_infectados_pct']:.1f}%** de la población se infectó. Para lograr inmunidad de rebaño se necesita inmunizar al **{metricas['umbral_rebano_pct']:.1f}%**."
                            )
                        else:
                            st.success(f"Con **R₀ = {r0_val:.2f} < 1**, la epidemia se extingue naturalmente.")

                    st.session_state["_sir_res"] = res_rk4
                    st.session_state["_sir_met"] = metricas
                    st.session_state["_sir_Nval"] = N_pop
                    st.session_state["_sir_tmaxval"] = int(t_max_sir)
                    st.session_state["_sir_dfval"] = df_sir

                except Exception as e:
                    st.error(f"Error en la simulación: {e}")

            # ── Visualizaciones persistentes ─────────────────────────────────
            if all(k in st.session_state for k in ["_sir_res", "_sir_met", "_sir_Nval", "_sir_tmaxval", "_sir_dfval"]):
                _res = st.session_state["_sir_res"]
                _N = st.session_state["_sir_Nval"]
                _tmax = st.session_state["_sir_tmaxval"]
                _df_sir = st.session_state["_sir_dfval"]
                _t_arr = _np.array(_res["t"])

                with st.container(border=True):
                    st.markdown("### 🎛️ Explorador Visual")
                    dia_sel = st.slider("Día", 0, _tmax, 0, 1, key="sir_sl")
                    _idx = int(_np.argmin(_np.abs(_t_arr - dia_sel)))
                    _s, _i, _r = _res["S"][_idx], _res["I"][_idx], _res["R"][_idx]
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("🔵 Susceptibles", f"{_s:,.0f}", f"{_s/_N*100:.1f}%")
                    mc2.metric("🔴 Infectados", f"{_i:,.0f}", f"{_i/_N*100:.1f}%")
                    mc3.metric("🟢 Recuperados", f"{_r:,.0f}", f"{_r/_N*100:.1f}%")
                    bd = pd.DataFrame([
                        {"G": "Susceptibles", "P": _s/_N*100, "o": 1},
                        {"G": "Infectados", "P": _i/_N*100, "o": 2},
                        {"G": "Recuperados", "P": _r/_N*100, "o": 3},
                    ])
                    st.altair_chart(alt.Chart(bd).mark_bar(cornerRadius=4, height=40).encode(
                        x=alt.X("P:Q", stack="zero", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="%")),
                        color=alt.Color("G:N", scale=alt.Scale(domain=["Susceptibles", "Infectados", "Recuperados"],
                            range=["#3b82f6", "#ef4444", "#22c55e"]), legend=alt.Legend(orient="bottom")),
                        order="o:Q", tooltip=["G", alt.Tooltip("P:Q", format=".1f")],
                    ).properties(height=60), use_container_width=True)

                with st.container(border=True):
                    st.markdown(f"### 🏘️ Población — Día {dia_sel}")
                    st.caption("🔵 Sana  🔴 Infectada  🟢 Recuperada")
                    _gn = min(_N, 2500)
                    _sc = _gn / _N
                    _ns = max(0, int(round(_s * _sc)))
                    _ni = max(0, int(round(_i * _sc)))
                    _nr = max(0, _gn - _ns - _ni)
                    _clrs = ["#3b82f6"] * _ns + ["#ef4444"] * _ni + ["#22c55e"] * _nr
                    _perm = _np.random.RandomState(42).permutation(len(_clrs))
                    _clrs = [_clrs[j] for j in _perm]
                    _cg = int(_np.ceil(_np.sqrt(_gn)))
                    df_g = pd.DataFrame({
                        "x": [j % _cg for j in range(len(_clrs))],
                        "y": [j // _cg for j in range(len(_clrs))], "c": _clrs,
                        "e": ["Sana" if c == "#3b82f6" else ("Infectada" if c == "#ef4444" else "Recuperada") for c in _clrs]
                    })
                    st.altair_chart(alt.Chart(df_g).mark_square(size=12).encode(
                        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None, sort="descending"),
                        color=alt.Color("c:N", scale=None), tooltip=["e"],
                    ).properties(height=350).configure_view(strokeWidth=0), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### 🎬 Animación de la Epidemia")
                    st.caption("Presioná ▶️ y mirá cómo el virus se propaga por la población día a día.")
                    _play = st.button("▶️ Reproducir Animación", key="sir_anim")
                    _ph_grid = st.empty()
                    _ph_bar = st.empty()
                    _ph_text = st.empty()
                    _nf = min(60, len(_res["t"]))
                    _fi = _np.linspace(0, len(_res["t"]) - 1, _nf, dtype=int)
                    _gn2 = min(_N, 1600)
                    _sc2 = _gn2 / _N
                    _cg2 = int(_np.ceil(_np.sqrt(_gn2)))
                    _base_perm = _np.random.RandomState(42).permutation(_gn2)
                    if _play:
                        for idx_f in _fi:
                            _tc = _res["t"][idx_f]
                            _sf = _res["S"][idx_f]
                            _if_ = _res["I"][idx_f]
                            _rf = _res["R"][idx_f]
                            _ns2 = max(0, int(round(_sf * _sc2)))
                            _ni2 = max(0, int(round(_if_ * _sc2)))
                            _nr2 = max(0, _gn2 - _ns2 - _ni2)
                            _clrs2 = ["#3b82f6"] * _ns2 + ["#ef4444"] * _ni2 + ["#22c55e"] * _nr2
                            _clrs2 = [_clrs2[j] for j in _base_perm[:len(_clrs2)]]
                            df_g2 = pd.DataFrame({
                                "x": [j % _cg2 for j in range(len(_clrs2))],
                                "y": [j // _cg2 for j in range(len(_clrs2))], "c": _clrs2
                            })
                            _ph_grid.altair_chart(alt.Chart(df_g2).mark_square(size=14).encode(
                                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None, sort="descending"),
                                color=alt.Color("c:N", scale=None),
                            ).properties(height=320, title=f"Día {_tc:.0f}").configure_view(strokeWidth=0), use_container_width=True)
                            _ad = pd.DataFrame([{"G": "Susceptibles", "P": _sf}, {"G": "Infectados", "P": _if_}, {"G": "Recuperados", "P": _rf}])
                            _ph_bar.altair_chart(alt.Chart(_ad).mark_bar(cornerRadius=6).encode(
                                x=alt.X("P:Q", scale=alt.Scale(domain=[0, _N]), axis=alt.Axis(format=",.0f")),
                                y=alt.Y("G:N", sort=["Susceptibles", "Infectados", "Recuperados"], axis=alt.Axis(title="")),
                                color=alt.Color("G:N", scale=alt.Scale(domain=["Susceptibles", "Infectados", "Recuperados"],
                                    range=["#3b82f6", "#ef4444", "#22c55e"]), legend=None),
                            ).properties(height=100), use_container_width=True)
                            _ph_text.markdown(f"### Día {_tc:.0f}\n🔵 Sanos: **{_sf:,.0f}** | 🔴 Infectados: **{_if_:,.0f}** | 🟢 Recuperados: **{_rf:,.0f}**")
                            _time.sleep(0.12)
                        st.success("✅ Animación completada")

                with st.container(border=True):
                    st.markdown("### 📊 Áreas Apiladas")
                    df_st = _df_sir.melt("Día", var_name="Grupo", value_name="Personas")
                    st.altair_chart(alt.Chart(df_st).mark_area().encode(
                        x=alt.X("Día:Q", title="Días"), y=alt.Y("Personas:Q", stack="zero"),
                        color=alt.Color("Grupo:N", scale=alt.Scale(domain=["Susceptibles", "Infectados", "Recuperados"],
                            range=["#3b82f6", "#ef4444", "#22c55e"])),
                        order=alt.Order("Grupo:N", sort="descending"),
                        tooltip=["Día", "Grupo", alt.Tooltip("Personas:Q", format=",.0f")],
                    ).properties(height=350).interactive(), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### 🔄 Diagrama de Fase (S vs I)")
                    df_ph = pd.DataFrame({"S": _res["S"], "I": _res["I"], "Día": _res["t"]})
                    _ln = alt.Chart(df_ph).mark_line(strokeWidth=2, color="#a855f7").encode(
                        x=alt.X("S:Q", title="Susceptibles"), y=alt.Y("I:Q", title="Infectados"), tooltip=["S", "I", "Día"]
                    ).properties(height=350)
                    df_ep = pd.DataFrame({"S": [_res["S"][0], _res["S"][-1]], "I": [_res["I"][0], _res["I"][-1]],
                        "l": ["Inicio", "Fin"], "c": ["#22d3ee", "#f97316"]})
                    _p2 = alt.Chart(df_ep).mark_point(size=120, filled=True, stroke="white", strokeWidth=1.5).encode(
                        x="S:Q", y="I:Q", color=alt.Color("c:N", scale=None), tooltip=["l", "S", "I"])
                    _lb = alt.Chart(df_ep).mark_text(dx=10, dy=-10, fontSize=11, fontWeight="bold").encode(
                        x="S:Q", y="I:Q", text="l:N", color=alt.Color("c:N", scale=None))
                    st.altair_chart((_ln + _p2 + _lb).interactive(), use_container_width=True)

        with side_col:
            _formulas_panel("Simulación SIR (Epidemia)")
            with st.container(border=True):
                st.markdown("### 💡 Ejemplos de β y γ")
                st.caption("**Gripe:** β ≈ 0.3, γ ≈ 0.14 (R₀ ≈ 2.1)")
                st.caption("**COVID-19:** β ≈ 0.4, γ ≈ 0.07 (R₀ ≈ 5.7)")
                st.caption("**Sarampión:** β ≈ 1.8, γ ≈ 0.14 (R₀ ≈ 13)")
                st.caption("**Enfermedad leve:** β ≈ 0.05, γ ≈ 0.1 (R₀ ≈ 0.5)")
                st.divider()
                st.markdown("### 🔑 Parámetros clave")
                st.caption("**β**: probabilidad de contagio por contacto × contactos diarios.")
                st.caption("**γ**: 1/γ = días promedio de enfermedad. Ej: γ=0.1 → 10 días enfermo.")
                st.caption("**R₀ = β/γ**: umbral epidémico. R₀ > 1 → brote.")
