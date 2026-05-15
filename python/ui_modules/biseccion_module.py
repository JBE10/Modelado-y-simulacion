"""biseccion_module.py — Bisección UI Module."""

import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, mostrar_resultados, _formulas_panel
from biseccion import biseccion


class BiseccionModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Biseccion"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Biseccion")
                st.caption("Metodo de intervalo con cambio de signo en [a,b].")
                with st.form("biseccion"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr = st.text_input("f(x)", "x**3 - x - 2")
                        a = st.number_input("a", value=1.0, format="%.10f")
                        b = st.number_input("b", value=2.0, format="%.10f")
                    with c2:
                        tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_bis")
                        mi = st.number_input("Max iter", value=100, min_value=1, step=1, key="mi_bis")
                    run = st.form_submit_button("Calcular", type="primary")

            if run:
                try:
                    f_fn = make_fn(expr)
                    res = biseccion(f_fn, a, b, tol=tol, max_iter=int(mi))
                    plot_cfg = st.session_state.get("plot_cfg", {"n_samples": 900, "expand_factor": 0.8})
                    mostrar_resultados(res, f_fn, "c", ["iter", "c", "f(c)", "error"], [a, b], plot_cfg)
                except Exception as e:
                    st.error(str(e))

        with side_col:
            _formulas_panel("Biseccion")
