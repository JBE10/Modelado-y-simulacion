"""newton_raphson_module.py — Newton-Raphson UI Module."""
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, mostrar_resultados, _formulas_panel, norm
from newton_raphson import newton_raphson


class NewtonRaphsonModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Newton-Raphson"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Newton-Raphson")
                st.caption("Metodo abierto con derivada explicita.")
                with st.form("newton"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "x**3 - x - 2", key="f_nr")
                        auto_deriv = st.checkbox("Calcular f'(x) automáticamente con sympy", value=False, key="auto_df")
                        expr_df = st.text_input("f'(x) manual (ignorado si auto está activo)", "3*x**2 - 1", key="df_nr")
                        x0 = st.number_input("x0", value=1.5, format="%.10f", key="x0_nr")
                    with c2:
                        tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_nr")
                        mi = st.number_input("Max iter", value=50, min_value=1, step=1, key="mi_nr")
                    run = st.form_submit_button("Calcular", type="primary")

            if run:
                if not expr_f.strip():
                    st.error("Debes ingresar f(x).")
                elif not auto_deriv and not expr_df.strip():
                    st.error("Debes ingresar f'(x) o activar el cálculo automático.")
                else:
                    try:
                        f_fn = make_fn(expr_f)
                        if auto_deriv:
                            import sympy
                            x_sym = sympy.Symbol("x")
                            MATH_SYMPY_NS = {
                                "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
                                "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                                "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
                                "sqrt": sympy.sqrt, "abs": sympy.Abs,
                                "cbrt": lambda z: sympy.root(z, 3),
                            }
                            parsed = sympy.sympify(norm(expr_f).replace("ln(", "log("), locals=MATH_SYMPY_NS)
                            derivada_sym = sympy.diff(parsed, x_sym)
                            st.info(f"f'(x) calculada automáticamente: **{derivada_sym}**")
                            st.latex(rf"f'(x) = {sympy.latex(derivada_sym)}")
                            df_fn = sympy.lambdify(x_sym, derivada_sym, "math")
                        else:
                            df_fn = make_fn(expr_df)
                        res = newton_raphson(f_fn, df_fn, x0, tol=tol, max_iter=int(mi))
                        plot_cfg = st.session_state.get("plot_cfg", {"n_samples": 900, "expand_factor": 0.8})
                        mostrar_resultados(
                            res, f_fn, "x_n+1",
                            ["iter", "x_n", "f(x_n)", "f'(x_n)", "x_n+1", "error"],
                            [min(x0, res["raiz"]) - 1, max(x0, res["raiz"]) + 1],
                            plot_cfg,
                        )
                    except Exception as e:
                        st.error(str(e))

        with side_col:
            _formulas_panel("Newton-Raphson")
