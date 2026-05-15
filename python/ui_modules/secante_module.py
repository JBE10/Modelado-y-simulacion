"""secante_module.py — Secante UI Module."""
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, mostrar_resultados, _formulas_panel
from secante import secante


class SecanteModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Secante"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Secante")
                st.caption("Metodo abierto sin derivada, usa dos semillas iniciales.")
                with st.form("secante"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "x**3 - x - 2", key="f_sec")
                        x0 = st.number_input("x0", value=1.0, format="%.10f", key="x0_sec")
                        x1 = st.number_input("x1", value=2.0, format="%.10f", key="x1_sec")
                    with c2:
                        tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_sec")
                        mi = st.number_input("Max iter", value=100, min_value=1, step=1, key="mi_sec")
                    run = st.form_submit_button("Calcular", type="primary")

            if run:
                if not expr_f.strip():
                    st.error("Debes ingresar f(x).")
                else:
                    try:
                        f_fn = make_fn(expr_f)
                        res = secante(f_fn, x0, x1, tol=tol, max_iter=int(mi))
                        plot_cfg = st.session_state.get("plot_cfg", {"n_samples": 900, "expand_factor": 0.8})
                        mostrar_resultados(
                            res, f_fn, "x_{n+1}",
                            ["iter", "x_n", "f(x_n)", "x_{n+1}", "error"],
                            [min(x0, x1, res["raiz"]) - 1, max(x0, x1, res["raiz"]) + 1],
                            plot_cfg,
                        )
                    except Exception as e:
                        st.error(str(e))

        with side_col:
            _formulas_panel("Secante")
