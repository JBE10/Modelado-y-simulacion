"""ecuaciones_diferenciales_module.py — EDO UI Module."""
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn_ty, make_fn_t, _formulas_panel, _fmt_iter_df
from runge_kutta import euler_method, rk2_method, rk4_method


class EdoModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Ecuaciones Diferenciales (EDO)"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Ecuaciones Diferenciales Ordinarias (PVI)")
                st.caption("Resuelve el Problema de Valor Inicial $y' = f(t, y)$ con $y(t_0) = y_0$.")
                with st.form("edo_form"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        expr_f = st.text_input("f(t, y)", "(y + t**2 - 2)", key="f_edo")
                        t0 = st.number_input("t₀", value=0.0, format="%.4f", key="t0_edo")
                        y0 = st.number_input("y₀", value=2.0, format="%.4f", key="y0_edo")
                    with c2:
                        tf = st.number_input("t_f", value=2.0, format="%.4f", key="tf_edo")
                        h = st.number_input("Tamaño de paso (h)", value=0.2, min_value=1e-5, format="%.5f", key="h_edo")
                        expr_exact = st.text_input("Solución Exacta y(t) (opcional)", "(t+1)**2 + exp(t) - t**2", help="Si se proporciona, se calculará el error global.")
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
                        f_exact = None
                        if expr_exact.strip():
                            try:
                                f_exact = make_fn_t(expr_exact)
                            except Exception as e:
                                st.warning(f"No se pudo procesar la solución exacta: {e}")

                        if metodo == "Comparar Todos":
                            res_euler = euler_method(f_fn_ty, t0, y0, tf, h)
                            res_rk2_heun = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Heun")
                            res_rk2_pm = rk2_method(f_fn_ty, t0, y0, tf, h, variante="Punto Medio")
                            res_rk4 = rk4_method(f_fn_ty, t0, y0, tf, h)
                            
                            data_list = [
                                pd.DataFrame({"t": res_euler["t_vals"], "y": res_euler["y_vals"], "Metodo": "Euler"}),
                                pd.DataFrame({"t": res_rk2_heun["t_vals"], "y": res_rk2_heun["y_vals"], "Metodo": "RK2 (Heun)"}),
                                pd.DataFrame({"t": res_rk2_pm["t_vals"], "y": res_rk2_pm["y_vals"], "Metodo": "RK2 (Punto Medio)"}),
                                pd.DataFrame({"t": res_rk4["t_vals"], "y": res_rk4["y_vals"], "Metodo": "RK4"}),
                            ]
                            
                            if f_exact:
                                t_dense = [t0 + i * (h/4) for i in range(int((tf-t0)/(h/4)) + 1)]
                                y_exact = [f_exact(t) for t in t_dense]
                                data_list.append(pd.DataFrame({"t": t_dense, "y": y_exact, "Metodo": "Exacta"}))

                            df_all = pd.concat(data_list, ignore_index=True)
                            
                            with st.container(border=True):
                                st.markdown("### Comparación de Métodos")
                                chart = alt.Chart(df_all).mark_line(point=True).encode(
                                    x=alt.X("t:Q", title="Tiempo (t)"),
                                    y=alt.Y("y:Q", title="Solución (y)", scale=alt.Scale(zero=False)),
                                    color=alt.Color("Metodo:N", scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#000000'])),
                                    strokeDash=alt.condition(alt.datum.Metodo == 'Exacta', alt.value([5, 5]), alt.value([0])),
                                    tooltip=["t", "y", "Metodo"]
                                ).properties(height=400).interactive()
                                st.altair_chart(chart, use_container_width=True)

                            if f_exact:
                                st.markdown("### Análisis de Error Global")
                                err_data = []
                                methods = [("Euler", res_euler), ("RK2 (Heun)", res_rk2_heun), ("RK2 (PM)", res_rk2_pm), ("RK4", res_rk4)]
                                for name, res_m in methods:
                                    for t_v, y_v in zip(res_m["t_vals"], res_m["y_vals"]):
                                        y_true = f_exact(t_v)
                                        err_data.append({"t": t_v, "Error Absoluto": abs(y_true - y_v), "Metodo": name})
                                
                                df_err = pd.DataFrame(err_data)
                                chart_err = alt.Chart(df_err).mark_line(point=True).encode(
                                    x=alt.X("t:Q", title="Tiempo (t)"),
                                    y=alt.Y("Error Absoluto:Q", title="Error Absoluto |y_true - y_aprox|", scale=alt.Scale(type='log' if df_err["Error Absoluto"].max() > 0 else 'linear')),
                                    color="Metodo:N",
                                    tooltip=["t", "Error Absoluto", "Metodo"]
                                ).properties(height=300).interactive()
                                st.altair_chart(chart_err, use_container_width=True)
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
                            if f_exact:
                                df_res["y_exacta"] = [f_exact(t) for t in res["t_vals"]]
                                df_res["error_abs"] = abs(df_res["y_exacta"] - df_res["y_n"])

                            with st.container(border=True):
                                c1r, c2r, c3r = st.columns(3)
                                c1r.metric("Método", res["metodo"])
                                c2r.metric(f"y({tf})", f"{df_res['y_n'].iloc[-1]:.6f}")
                                if f_exact:
                                    c3r.metric("Error Final", f"{df_res['error_abs'].iloc[-1]:.2e}")

                            with st.container(border=True):
                                st.markdown("### Gráfica de la Solución")
                                base = alt.Chart(df_res).encode(x=alt.X("t_n:Q", title="Tiempo (t)"))
                                line_aprox = base.mark_line(point=True, color="#d946ef").encode(y=alt.Y("y_n:Q", title="y"), tooltip=["t_n", "y_n"])
                                if f_exact:
                                    line_exact = base.mark_line(color="#000", strokeDash=[5, 5]).encode(y="y_exacta:Q", tooltip=["t_n", "y_exacta"])
                                    chart = alt.layer(line_exact, line_aprox).properties(height=350).interactive()
                                else:
                                    chart = line_aprox.properties(height=350).interactive()
                                st.altair_chart(chart, use_container_width=True)
                                
                            with st.container(border=True):
                                st.markdown("### Tabla de Iteraciones")
                                st.dataframe(_fmt_iter_df(df_res, decimals=6), hide_index=True, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error en el cálculo: {e}")

        with side_col:
            _formulas_panel("Ecuaciones Diferenciales (EDO)")
