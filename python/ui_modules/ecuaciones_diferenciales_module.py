"""ecuaciones_diferenciales_module.py — EDO UI Module."""
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn_ty, _formulas_panel, _fmt_iter_df
from runge_kutta import euler_method, rk2_method, rk4_method


class EdoModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Ecuaciones Diferenciales (EDO)"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Ecuaciones Diferenciales Ordinarias (PVI)")
                st.caption("Resuelve el Problema de Valor Inicial $y' = f(t, y)$ con $y(t_0) = y_0$.")
                with st.form("edo_form"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        expr_f = st.text_input("f(t, y)", "y - t**2 + 1", key="f_edo")
                        t0 = st.number_input("t₀", value=0.0, format="%.4f", key="t0_edo")
                        y0 = st.number_input("y₀", value=0.5, format="%.4f", key="y0_edo")
                    with c2:
                        tf = st.number_input("t_f", value=2.0, format="%.4f", key="tf_edo")
                        h = st.number_input("Tamaño de paso (h)", value=0.2, min_value=1e-5, format="%.5f", key="h_edo")
                    with c3:
                        metodo = st.selectbox("Método", ["Euler", "RK2 (Punto Medio)", "RK2 (Heun)", "RK2 (Ralston)", "RK4", "Comparar Todos"])
                    run_edo = st.form_submit_button("Resolver", type="primary")

            if run_edo:
                if not expr_f.strip():
                    st.error("Debes ingresar f(t, y).")
                elif tf <= t0:
                    st.error("El tiempo final (t_f) debe ser mayor a t₀.")
                else:
                    try:
                        f_fn_ty = make_fn_ty(expr_f)
                        try:
                            f_fn_ty(t0, y0)
                        except Exception:
                            st.error("Error evaluando la función. Asegúrate de usar variables 't' e 'y'.")
                            raise

                        if metodo == "Comparar Todos":
                            res_euler = euler_method(f_fn_ty, t0, y0, tf, h)
                            res_rk2_heun = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Heun")
                            res_rk2_pm = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Punto Medio")
                            res_rk4 = rk4_method(f_fn_ty, t0, y0, tf, h)
                            df_all = pd.concat([
                                pd.DataFrame({"t": res_euler["t_vals"], "y": res_euler["y_vals"], "Metodo": "Euler"}),
                                pd.DataFrame({"t": res_rk2_heun["t_vals"], "y": res_rk2_heun["y_vals"], "Metodo": "RK2 (Heun)"}),
                                pd.DataFrame({"t": res_rk2_pm["t_vals"], "y": res_rk2_pm["y_vals"], "Metodo": "RK2 (Punto Medio)"}),
                                pd.DataFrame({"t": res_rk4["t_vals"], "y": res_rk4["y_vals"], "Metodo": "RK4"}),
                            ], ignore_index=True)
                            with st.container(border=True):
                                st.markdown("### Comparación de Métodos")
                                chart = alt.Chart(df_all).mark_line(point=True).encode(
                                    x=alt.X("t:Q", title="Tiempo (t)"),
                                    y=alt.Y("y:Q", title="Solución (y)"),
                                    color="Metodo:N", tooltip=["t", "y", "Metodo"]
                                ).properties(height=400).interactive()
                                st.altair_chart(chart, use_container_width=True)
                        else:
                            if metodo == "Euler":
                                res = euler_method(f_fn_ty, t0, y0, tf, h)
                            elif metodo == "RK2 (Punto Medio)":
                                res = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Punto Medio")
                            elif metodo == "RK2 (Heun)":
                                res = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Heun")
                            elif metodo == "RK2 (Ralston)":
                                res = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Ralston")
                            elif metodo == "RK4":
                                res = rk4_method(f_fn_ty, t0, y0, tf, h)

                            df_res = pd.DataFrame({"t_n": res["t_vals"], "y_n": res["y_vals"]})
                            with st.container(border=True):
                                c1r, c2r = st.columns(2)
                                c1r.metric("Método Utilizado", res["metodo"])
                                c2r.metric(f"Valor Final y({tf})", f"{df_res['y_n'].iloc[-1]:.6f}")
                            with st.container(border=True):
                                st.markdown("### Gráfica de la Solución")
                                chart = alt.Chart(df_res).mark_line(point=True, color="#d946ef").encode(
                                    x=alt.X("t_n:Q", title="Tiempo (t)"),
                                    y=alt.Y("y_n:Q", title="Solución (y)"),
                                    tooltip=["t_n", "y_n"]
                                ).properties(height=350).interactive()
                                st.altair_chart(chart, use_container_width=True)
                            with st.container(border=True):
                                st.markdown("### Tabla de Iteraciones")
                                st.dataframe(_fmt_iter_df(df_res, decimals=6), hide_index=True, use_container_width=True)

                    except Exception as e:
                        if "t_n" not in str(e):
                            st.error(str(e))

        with side_col:
            _formulas_panel("Ecuaciones Diferenciales (EDO)")
