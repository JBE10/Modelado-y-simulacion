"""integracion_numerica_module.py — Integración Numérica UI Module."""
import math
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, _formulas_panel, parse_number_cell, expr_integrando_latex, _latex_bound
from integracion_numerica import (
    integracion_rectangulo,
    integracion_trapecio,
    integracion_simpson_13,
    integracion_simpson_38,
    integral_referencia,
)


class IntegracionNumericaModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Integración Numérica"

    def render(self, **kwargs):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Integración Numérica (Newton-Cotes)")
                st.caption("Estima el área bajo la curva f(x) en el intervalo [a, b].")
                with st.form("integracion"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "x**2 * exp(-x)", key="int_f")
                        str_a = st.text_input("a (Límite inferior)", "0", key="int_a")
                        str_b = st.text_input("b (Límite superior)", "4", key="int_b")
                    with c2:
                        metodo = st.selectbox("Método", ["Rectángulo (Medio)", "Trapecio", "Simpson 1/3", "Simpson 3/8"])
                        n = st.number_input("n (Subintervalos)", value=12, min_value=1, step=1, key="int_n")
                        ndec = st.number_input("Decimales a mostrar", min_value=6, max_value=24, value=16, step=1, key="int_ndec")
                    run_int = st.form_submit_button("Calcular Integral", type="primary")

            if run_int:
                if not expr_f.strip():
                    st.error("Debes ingresar f(x).")
                else:
                    try:
                        a = parse_number_cell(str_a)
                        b = parse_number_cell(str_b)
                        if b <= a:
                            st.error("El límite superior 'b' debe ser mayor que el inferior 'a'.")
                        else:
                            f_fn = make_fn(expr_f)
                            res = None
                            if metodo == "Rectángulo (Medio)":
                                res = integracion_rectangulo(f_fn, a, b, int(n))
                            elif metodo == "Trapecio":
                                res = integracion_trapecio(f_fn, a, b, int(n))
                            elif metodo == "Simpson 1/3":
                                res = integracion_simpson_13(f_fn, a, b, int(n))
                            elif metodo == "Simpson 3/8":
                                res = integracion_simpson_38(f_fn, a, b, int(n))

                            if res:
                                nd = int(ndec)
                                I_ref, err_quad = None, None
                                try:
                                    I_ref, err_quad = integral_referencia(f_fn, a, b)
                                except Exception as qe:
                                    st.warning(f"No se pudo obtener la integral de referencia (quad): {qe}")

                                Lg = expr_integrando_latex(expr_f)
                                ta = _latex_bound(a, nd)
                                tb = _latex_bound(b, nd)
                                fmt_f = f"{{:.{nd}f}}"

                                with st.container(border=True):
                                    st.markdown("### Integral en LaTeX")
                                    rows = [
                                        rf"\int_{{{ta}}}^{{{tb}}} \left({Lg}\right) \,\mathrm{{d}}x "
                                        rf"&\approx {fmt_f.format(res['valor'])}"
                                    ]
                                    if I_ref is not None:
                                        rows.append(rf"I_{{\mathrm{{ref}}}} &\approx {fmt_f.format(I_ref)}")
                                        e_trunc = I_ref - res["valor"]
                                        rows.append(rf"E_{{\mathrm{{trunc}}}} = I_{{\mathrm{{ref}}}} - I_{{\mathrm{{aprox}}}} &\approx {fmt_f.format(e_trunc)}")
                                        rows.append(rf"\left| E_{{\mathrm{{trunc}}}} \right| &\approx {fmt_f.format(abs(e_trunc))}")
                                    if err_quad is not None and math.isfinite(err_quad):
                                        rows.append(rf"\varepsilon_{{\mathrm{{quad}}}} &\approx {fmt_f.format(err_quad)}")
                                    st.latex("\\begin{aligned} " + " \\\\ ".join(rows) + " \\end{aligned}")

                                with st.container(border=True):
                                    st.markdown(f"### {res['metodo']}")
                                    cols = st.columns(4)
                                    cols[0].metric("∫ f (referencia)", fmt_f.format(I_ref) if I_ref is not None else "—")
                                    cols[1].metric("Integral aproximada", fmt_f.format(res["valor"]))
                                    cols[2].metric("h (paso)", f"{res['h']:.{min(nd, 10)}f}")
                                    cols[3].metric("Nodos", len(res["x_vals"]))

                                if I_ref is not None:
                                    e_trunc = I_ref - res["valor"]
                                    c_err = st.columns(3)
                                    c_err[0].metric("Error truncamiento |E|", fmt_f.format(abs(e_trunc)))
                                    c_err[1].metric("E_trunc (con signo)", f"{e_trunc:+.{nd}f}")
                                    c_err[2].metric("ε_quad (referencia)",
                                        fmt_f.format(err_quad) if err_quad is not None and math.isfinite(err_quad) else "—")

                                with st.container(border=True):
                                    st.markdown("### Área de Aproximación")
                                    xs = np.linspace(a, b, 600)
                                    ys = []
                                    for xv in xs:
                                        try: ys.append(float(f_fn(xv)))
                                        except: ys.append(float("nan"))
                                    df_smooth = pd.DataFrame({"x": xs, "f(x)": ys})
                                    curva = alt.Chart(df_smooth).mark_line(color="#2563eb", strokeWidth=3).encode(
                                        x=alt.X("x:Q", title="x"), y=alt.Y("f(x):Q", title="f(x)"), tooltip=["x", "f(x)"])
                                    df_approx = pd.DataFrame({"x": res["x_vals"], "f(x)": res["y_vals"]})
                                    if metodo == "Rectángulo (Medio)":
                                        aprox = alt.Chart(df_approx).mark_bar(opacity=0.4, color="#ef4444", size=(1000/int(n))*0.5).encode(x="x:Q", y="f(x):Q")
                                    else:
                                        aprox = alt.Chart(df_approx).mark_area(opacity=0.4, color="#ef4444").encode(x="x:Q", y="f(x):Q")
                                    px = alt.Chart(df_approx).mark_point(color="#dc2626", filled=True, size=60).encode(x="x:Q", y="f(x):Q")
                                    st.altair_chart((curva + aprox + px).properties(height=350).interactive(), use_container_width=True)

                    except Exception as e:
                        st.error(str(e))

        with side_col:
            _formulas_panel("Integración Numérica")
