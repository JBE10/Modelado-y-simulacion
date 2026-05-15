"""diferencias_finitas_module.py — Diferencias Finitas UI Module."""
import math
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, _formulas_panel, _fmt_iter_df, parse_number_cell
from diferencias_finitas import (
    diferencias_finitas_funcion,
    diferencias_finitas_tabla,
    interpolacion_newton_divididas,
)


class DiferenciasFinitasModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Diferencias Finitas"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.expander("📐 Ver fórmulas de Diferencias Finitas Centradas", expanded=True):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown("**1ra Derivada — Centrada** ⭐ (O(h²))")
                    st.latex(r"f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{2h}")
                    st.markdown("**2da Derivada — Centrada** ⭐ (O(h²))")
                    st.latex(r"f''(x_i) \approx \frac{f(x_{i+1}) - 2\,f(x_i) + f(x_{i-1})}{h^2}")
                with col_f2:
                    st.markdown("**Extremo izquierdo — Progresiva** (O(h))")
                    st.latex(r"f'(x_0) \approx \frac{f(x_1) - f(x_0)}{h}")
                    st.latex(r"f''(x_0) \approx \frac{f(x_2) - 2\,f(x_1) + f(x_0)}{h^2}")
                    st.markdown("**Extremo derecho — Regresiva** (O(h))")
                    st.latex(r"f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{h}")
                    st.latex(r"f''(x_n) \approx \frac{f(x_n) - 2\,f(x_{n-1}) + f(x_{n-2})}{h^2}")

            modo = st.radio("Modo", [
                "Función continua (ingreso f(x) y h)",
                "Tabla de datos discretos",
                "Interpolación Newton — diferencias divididas (cualquier espaciado en x)",
            ], horizontal=True)

            if modo == "Función continua (ingreso f(x) y h)":
                self._modo_funcion()
            elif modo == "Tabla de datos discretos":
                self._modo_tabla()
            else:
                self._modo_newton()

        with side_col:
            _formulas_panel("Diferencias Finitas")

    def _modo_funcion(self):
        with st.container(border=True):
            st.subheader("Diferencias Finitas — Función continua")
            st.caption("Estima f' y f'' en un punto x₀ con paso h usando diferencias centradas.")
            with st.form("dif_fin_fn"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f = st.text_input("f(x)", "x**3 - 2*x + 1", key="df_f")
                    expr_d1 = st.text_input("f' exacta (opcional)", "3*x**2 - 2", key="df_d1")
                    expr_d2 = st.text_input("f'' exacta (opcional)", "6*x", key="df_d2")
                with c2:
                    x0_val = st.number_input("x₀", value=2.0, format="%.6f", key="df_x0")
                    h_val = st.number_input("h (paso)", value=0.1, min_value=1e-10, format="%.8f", key="df_h")
                    n_h = st.slider("Comparar N valores de h", 1, 20, 8, key="df_nh")
                run_fn = st.form_submit_button("Calcular", type="primary")

        if run_fn:
            try:
                f_fn = make_fn(expr_f)
                d1_fn = make_fn(expr_d1) if expr_d1.strip() else None
                d2_fn = make_fn(expr_d2) if expr_d2.strip() else None
                exacta_d1 = float(d1_fn(x0_val)) if d1_fn else None
                exacta_d2 = float(d2_fn(x0_val)) if d2_fn else None

                res = diferencias_finitas_funcion(f_fn, x0_val, h_val, exacta_d1=exacta_d1, exacta_d2=exacta_d2)

                with st.container(border=True):
                    st.markdown("### Resultado en x₀")
                    cols = st.columns(4)
                    cols[0].metric("f(x₀)", f"{res['f(x0)']:.10f}")
                    cols[1].metric("f'(x₀) centrada", f"{res['d1']:.10f}")
                    cols[2].metric("f''(x₀) centrada", f"{res['d2']:.10f}")
                    cols[3].metric("h usado", f"{h_val:.2e}")
                    if exacta_d1 is not None:
                        st.caption(f"Valor exacto f'(x₀) = {exacta_d1:.10f}  →  Error abs = {res['error_d1']:.8e}")
                    if exacta_d2 is not None:
                        st.caption(f"Valor exacto f''(x₀) = {exacta_d2:.10f}  →  Error abs = {res['error_d2']:.8e}")
                    st.latex(r"f'(x_0) \approx \frac{f(x_0+h)\,-\,f(x_0-h)}{2h}")
                    st.latex(r"f''(x_0) \approx \frac{f(x_0+h)\,-\,2f(x_0)\,+\,f(x_0-h)}{h^2}")

                with st.container(border=True):
                    st.markdown("### Evaluaciones en x₀")
                    df_eval = pd.DataFrame([
                        {"Punto": "x₀ - h", "x": x0_val - h_val, "f(x)": res["f(x0-h)"]},
                        {"Punto": "x₀",     "x": x0_val,          "f(x)": res["f(x0)"]},
                        {"Punto": "x₀ + h", "x": x0_val + h_val,  "f(x)": res["f(x0+h)"]},
                    ])
                    st.dataframe(_fmt_iter_df(df_eval), hide_index=True, use_container_width=True)

                if exacta_d1 is not None or exacta_d2 is not None:
                    with st.container(border=True):
                        st.markdown("### Convergencia del error según h")
                        hs = [h_val * (0.5 ** k) for k in range(n_h)]
                        rows_err = []
                        for hk in hs:
                            rk = diferencias_finitas_funcion(f_fn, x0_val, hk, exacta_d1=exacta_d1, exacta_d2=exacta_d2)
                            rows_err.append({
                                "h": hk, "f'(x₀)": rk["d1"], "f''(x₀)": rk["d2"],
                                "error f'": rk.get("error_d1"), "error f''": rk.get("error_d2"),
                            })
                        df_err = pd.DataFrame(rows_err)
                        st.dataframe(df_err.style.format({"h": "{:.2e}", "f'(x₀)": "{:.10f}",
                            "f''(x₀)": "{:.10f}", "error f'": "{:.4e}", "error f''": "{:.4e}"}),
                            hide_index=True, use_container_width=True)
                        err_cols = []
                        if exacta_d1 is not None: err_cols.append("error f'")
                        if exacta_d2 is not None: err_cols.append("error f''")
                        df_melt = df_err[["h"] + err_cols].melt("h", var_name="derivada", value_name="error")
                        chart_err = (alt.Chart(df_melt).mark_line(point=True).encode(
                            x=alt.X("h:Q", title="h", scale=alt.Scale(type="log")),
                            y=alt.Y("error:Q", title="Error absoluto", scale=alt.Scale(type="log")),
                            color="derivada:N", tooltip=["h", "derivada", "error"]
                        ).properties(height=300, title="Error vs h (escala log-log)"))
                        st.altair_chart(chart_err, use_container_width=True)
                        st.caption("La pendiente ~2 en log-log confirma el orden O(h²) de las diferencias centradas.")

                with st.container(border=True):
                    st.markdown("### Visualización de f(x) y los puntos usados")
                    xa = x0_val - abs(h_val) * 10
                    xb = x0_val + abs(h_val) * 10
                    paso_g = (xb - xa) / 600
                    datos_g = []
                    for k in range(601):
                        xv = xa + k * paso_g
                        try:
                            yv = float(f_fn(xv))
                            if math.isfinite(yv):
                                datos_g.append({"x": xv, "y": yv})
                        except Exception:
                            pass
                    df_g = pd.DataFrame(datos_g)
                    curva_g = alt.Chart(df_g).mark_line(color="#3b82f6", strokeWidth=2).encode(x="x:Q", y="y:Q", tooltip=["x", "y"])
                    pts_data = pd.DataFrame([
                        {"x": x0_val - h_val, "y": res["f(x0-h)"], "label": "x₀-h"},
                        {"x": x0_val,         "y": res["f(x0)"],   "label": "x₀"},
                        {"x": x0_val + h_val, "y": res["f(x0+h)"], "label": "x₀+h"},
                    ])
                    puntos_g = alt.Chart(pts_data).mark_point(size=150, filled=True).encode(
                        x="x:Q", y="y:Q",
                        color=alt.Color("label:N", legend=alt.Legend(title="Punto")),
                        tooltip=["label", "x", "y"],
                    )
                    st.altair_chart((curva_g + puntos_g).properties(height=320, title="f(x) con puntos de diferenciación").interactive(), use_container_width=True)
            except Exception as e:
                st.error(str(e))

    def _modo_tabla(self):
        with st.container(border=True):
            st.subheader("Diferencias Finitas — Tabla de datos discretos")
            st.caption("Pares x,y (coma decimal o expresiones: pi, pi/2, sqrt(2), etc.). Interior → centrada | Extremos → progresiva/regresiva.")
            with st.form("dif_fin_tabla"):
                puntos_raw = st.text_area("Puntos (x, y) — uno por línea o separados por ;",
                    value="0, 1\n1, 2\n2, 0\n3, 2\n4, 3", height=160, key="df_tabla")
                run_tab = st.form_submit_button("Calcular", type="primary")

        if run_tab:
            try:
                lineas = [l.strip() for l in puntos_raw.replace(";", "\n").splitlines() if l.strip()]
                xs_t, ys_t = [], []
                for linea in lineas:
                    partes = linea.split(",")
                    if len(partes) != 2:
                        raise ValueError(f"Formato incorrecto en línea: '{linea}'")
                    xs_t.append(parse_number_cell(partes[0]))
                    ys_t.append(parse_number_cell(partes[1]))
                res_tab = diferencias_finitas_tabla(xs_t, ys_t)
                with st.container(border=True):
                    st.markdown("### Tabla de derivadas")
                    df_tab = pd.DataFrame(res_tab["filas"]).rename(columns={
                        "i": "i", "x_i": "x_i", "y_i": "y_i (f)",
                        "f'(x_i)": "f'(x_i)", "tipo_d1": "Tipo 1ra der",
                        "f''(x_i)": "f''(x_i)", "tipo_d2": "Tipo 2da der",
                    })
                    st.dataframe(df_tab.style.applymap(
                        lambda v: "color: #86efac; font-weight:600" if v == "Centrada" else
                                  ("color: #fbbf24" if v in ("Progresiva", "Regresiva") else ""),
                        subset=["Tipo 1ra der", "Tipo 2da der"],
                    ), hide_index=True, use_container_width=True)
                    st.caption("🟢 Centrada (O(h²))  🟡 Progresiva/Regresiva (O(h))")
                with st.container(border=True):
                    st.markdown("### Gráficas")
                    df_plot = pd.DataFrame(res_tab["filas"])
                    g1, g2 = st.columns(2)
                    with g1:
                        st.altair_chart(alt.Chart(df_plot).mark_line(point=True, color="#3b82f6").encode(
                            x=alt.X("x_i:Q", title="x"), y=alt.Y("y_i:Q", title="f(x)"), tooltip=["x_i", "y_i"]
                        ).properties(height=270, title="Datos f(x)"), use_container_width=True)
                    with g2:
                        df_d1 = df_plot[["x_i", "f'(x_i)", "tipo_d1"]].copy()
                        st.altair_chart(alt.Chart(df_d1).mark_line(point=True).encode(
                            x=alt.X("x_i:Q", title="x"), y=alt.Y("f'(x_i):Q", title="f'(x)"),
                            color=alt.Color("tipo_d1:N", title="Esquema"), tooltip=["x_i", "f'(x_i)", "tipo_d1"]
                        ).properties(height=270, title="Primera derivada estimada"), use_container_width=True)
            except Exception as e:
                st.error(str(e))

    def _modo_newton(self):
        with st.container(border=True):
            st.subheader("Interpolación de Newton (diferencias divididas)")
            st.caption("x distintos en orden creciente; el paso puede variar.")
            with st.expander("Fórmulas", expanded=False):
                st.latex(r"f[x_i,\ldots,x_{i+k}]=\frac{f[x_{i+1},\ldots,x_{i+k}]-f[x_i,\ldots,x_{i+k-1}]}{x_{i+k}-x_i}")
                st.latex(r"P(x)=f[x_0]+f[x_0,x_1](x-x_0)+f[x_0,x_1,x_2](x-x_0)(x-x_1)+\cdots")
            with st.form("dif_newton_interp"):
                puntos_n = st.text_area("Puntos (x, y) — una fila por nodo; expresiones: pi, pi/4, sqrt(2)/2, …",
                    value="0, 0\npi/4, sqrt(2)/2\npi/2, 1\npi, 0", height=140, key="newton_xy")
                x_eval_str = st.text_input("Evaluar P(x) en x = (opcional)", "")
                run_n = st.form_submit_button("Calcular polinomio", type="primary")

        if run_n:
            try:
                lineas = [l.strip() for l in puntos_n.replace(";", "\n").splitlines() if l.strip()]
                xs_n, ys_n, incognitas = [], [], []
                for linea in lineas:
                    partes = linea.split(",")
                    if len(partes) != 2:
                        raise ValueError(f"Formato incorrecto en línea: '{linea}'. Usa 'x, y'.")
                    xv = parse_number_cell(partes[0])
                    y_str = partes[1].strip()
                    try:
                        ys_n.append(parse_number_cell(y_str)); xs_n.append(xv)
                    except ValueError:
                        incognitas.append((xv, y_str))

                if len(xs_n) < 2:
                    raise ValueError(f"Se necesitan al menos dos nodos numéricos (tienes {len(xs_n)}).")

                res_n = interpolacion_newton_divididas(xs_n, ys_n)

                with st.container(border=True):
                    st.markdown("### Tabla de diferencias divididas")
                    c1, c2 = st.columns(2)
                    c1.metric("x₀ (primer nodo)", f"{res_n['x0']:.10g}")
                    c2.metric("Nodos equiespaciados", "sí" if res_n["nodos_equiespaciados"] else "no")
                    st.dataframe(_fmt_iter_df(pd.DataFrame(res_n["filas_tabla"])), hide_index=True, use_container_width=True)

                with st.container(border=True):
                    st.markdown("### Polinomio (forma Newton en factores)")
                    st.latex(r"P(x) = " + res_n["latex_newton_divididas"])
                    st.markdown("### Polinomio en x (expandido, potencias)")
                    st.latex(r"P(x) = " + res_n["latex_polinomio"])
                    st.markdown("### Polinomio en Python (para copiar a Función)")
                    st.text_area("Expresión en Python", value=res_n.get("python_polinomio", ""), height=68, key="python_copy_newton")

                coefs = res_n["coefs_potencias"]
                def _eval_poly(xv):
                    return sum(coefs[i] * (xv ** i) for i in range(len(coefs)))

                pts_incognitas_grafica = []
                for _, (xv, nombre) in enumerate(incognitas):
                    val_pred = _eval_poly(xv)
                    st.success(f"**Incógnita resuelta:** Al evaluar $P({xv})$, obtenemos que ${nombre} \\approx {val_pred:.15g}$")
                    pts_incognitas_grafica.append({"x": xv, "y": val_pred, "tipo": f"Incógnita ({nombre})"})

                if x_eval_str.strip():
                    try:
                        val_x_eval = parse_number_cell(x_eval_str)
                        val_p = _eval_poly(val_x_eval)
                        st.info(f"**Evaluación manual:** $P({val_x_eval}) \\approx {val_p:.15g}$")
                    except Exception as e:
                        st.warning(f"No se pudo evaluar x = {x_eval_str}: {e}")

                with st.container(border=True):
                    st.markdown("### Verificación en los nodos")
                    ver_rows = [{"x": xv, "y dado": yv, "P(x)": _eval_poly(xv), "error": abs(_eval_poly(xv) - yv)}
                                for xv, yv in zip(xs_n, ys_n)]
                    st.dataframe(pd.DataFrame(ver_rows), hide_index=True, use_container_width=True)

                xa_n, xb_n = min(xs_n), max(xs_n)
                marg = max((xb_n - xa_n) * 0.25, 0.5)
                npt = 300
                xs_plot = [xa_n - marg + (k / max(npt - 1, 1)) * (xb_n - xa_n + 2 * marg) for k in range(npt)]
                ys_plot = [_eval_poly(xv) for xv in xs_plot]
                df_curve = pd.DataFrame({"x": xs_plot, "y": ys_plot})
                df_pts_conocidos = pd.DataFrame([{"x": x, "y": y, "tipo": "Dato conocido"} for x, y in zip(xs_n, ys_n)])
                df_pts_totales = pd.concat([df_pts_conocidos, pd.DataFrame(pts_incognitas_grafica)]) if pts_incognitas_grafica else df_pts_conocidos
                curva_p = alt.Chart(df_curve).mark_line(color="#7c3aed", strokeWidth=2).encode(x="x:Q", y="y:Q")
                puntos_p = alt.Chart(df_pts_totales).mark_point(size=150, filled=True).encode(
                    x="x:Q", y="y:Q",
                    color=alt.Color("tipo:N", scale=alt.Scale(domain=["Dato conocido", "Incógnita (k)"], range=["#f97316", "#10b981"])),
                    tooltip=["x", "y", "tipo"]
                )
                with st.container(border=True):
                    st.markdown("### P(x) y nodos")
                    st.altair_chart((curva_p + puntos_p).properties(height=320).interactive(), use_container_width=True)
            except Exception as e:
                st.error(str(e))
