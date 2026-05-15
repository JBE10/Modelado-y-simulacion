"""comparar_raices_module.py — Comparar Raíces UI Module."""
import math
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, norm
from biseccion import biseccion
from newton_raphson import newton_raphson
from secante import secante


class CompararRaicesModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Comparar Raíces"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Comparador de Métodos de Raíces")
                st.caption("Ejecuta Bisección, Newton-Raphson y Secante sobre la misma f(x).")
                with st.form("comparar_raices"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "x**3 - x - 2", key="cmp_f")
                        expr_df = st.text_input("f'(x) (vacío = auto)", "", key="cmp_df")
                        a_cmp = st.number_input("a (Bisección)", value=1.0, format="%.10f", key="cmp_a")
                        b_cmp = st.number_input("b (Bisección)", value=2.0, format="%.10f", key="cmp_b")
                    with c2:
                        x0_cmp = st.number_input("x0 (Newton/Secante)", value=1.0, format="%.10f", key="cmp_x0")
                        x1_cmp = st.number_input("x1 (Secante)", value=2.0, format="%.10f", key="cmp_x1")
                        tol_cmp = st.number_input("Tolerancia", value=1e-10, min_value=1e-15, format="%.15f", key="cmp_tol")
                        mi_cmp = st.number_input("Max iter", value=100, min_value=1, step=1, key="cmp_mi")
                    run_cmp = st.form_submit_button("Comparar", type="primary")

            if run_cmp and expr_f.strip():
                try:
                    f_fn = make_fn(expr_f)
                    resultados_cmp = {}
                    try:
                        resultados_cmp["Bisección"] = biseccion(f_fn, a_cmp, b_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    except Exception as e:
                        st.warning(f"Bisección falló: {e}")
                    try:
                        if expr_df.strip():
                            df_fn = make_fn(expr_df)
                        else:
                            import sympy
                            x_sym = sympy.Symbol("x")
                            ns = {"pi": sympy.pi, "e": sympy.E, "E": sympy.E,
                                  "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                                  "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
                                  "sqrt": sympy.sqrt, "abs": sympy.Abs}
                            parsed = sympy.sympify(norm(expr_f).replace("ln(", "log("), locals=ns)
                            df_fn = sympy.lambdify(x_sym, sympy.diff(parsed, x_sym), "math")
                        resultados_cmp["Newton-Raphson"] = newton_raphson(f_fn, df_fn, x0_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    except Exception as e:
                        st.warning(f"Newton-Raphson falló: {e}")
                    try:
                        resultados_cmp["Secante"] = secante(f_fn, x0_cmp, x1_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    except Exception as e:
                        st.warning(f"Secante falló: {e}")

                    if resultados_cmp:
                        with st.container(border=True):
                            st.markdown("### Tabla Comparativa")
                            rows = []
                            for nombre, r in resultados_cmp.items():
                                err = abs(f_fn(r["raiz"])) if r["raiz"] is not None else None
                                rows.append({"Método": nombre, "Raíz": f"{r['raiz']:.12f}",
                                             "Iteraciones": r["iteraciones"],
                                             "|f(raíz)|": f"{err:.4e}" if err else "N/A",
                                             "Convergió": "✅" if r["convergio"] else "❌"})
                            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                        with st.container(border=True):
                            st.markdown("### Convergencia Comparada (log₁₀ error)")
                            all_s = []
                            for nombre, r in resultados_cmp.items():
                                for row in r["historial"]:
                                    if row.get("error", 0) > 0:
                                        all_s.append({"iter": row["iter"],
                                                      "log₁₀(error)": math.log10(row["error"]),
                                                      "Método": nombre})
                            if all_s:
                                chart = alt.Chart(pd.DataFrame(all_s)).mark_line(point=True).encode(
                                    x="iter:Q", y="log₁₀(error):Q",
                                    color="Método:N", tooltip=["iter", "Método", "log₁₀(error)"]
                                ).properties(height=350, title="Velocidad de Convergencia por Método")
                                st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(str(e))

        with side_col:
            st.markdown("### Comparador")
            st.caption("Ejecuta los 3 métodos sobre la misma función y compara convergencia y precisión.")
            st.latex(r"p \approx \frac{\ln|e_{n+1}/e_n|}{\ln|e_n/e_{n-1}|}")
            st.caption("p≈1: Bisección | p≈1.62: Secante | p≈2: Newton")
