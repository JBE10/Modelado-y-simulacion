"""punto_fijo_module.py — Punto Fijo UI Module."""
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, mostrar_resultados, _formulas_panel, _fmt_iter_df
from punto_fijo import punto_fijo


class PuntoFijoModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Punto Fijo"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Punto Fijo")
                st.caption("Define una transformacion g(x) para iterar x_(n+1)=g(x_n).")
                with st.form("punto_fijo"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "x**3 - x - 2")
                        expr_g = st.text_input("g(x)", "(x + 2)**(1/3)")
                        x0 = st.number_input("x0", value=1.5, format="%.10f")
                    with c2:
                        tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_pf")
                        mi = st.number_input("Max iter", value=100, min_value=1, step=1, key="mi_pf")
                        usar_aitken = st.checkbox("Aitken", value=False)
                    run = st.form_submit_button("Calcular", type="primary")

            if run:
                if not expr_f.strip() or not expr_g.strip():
                    st.error("Debes ingresar f(x) y g(x).")
                else:
                    try:
                        f_fn = make_fn(expr_f)
                        g_fn = make_fn(expr_g)
                        res = punto_fijo(g_fn, x0, tol=tol, max_iter=int(mi), f=f_fn)
                        plot_cfg = st.session_state.get("plot_cfg", {"n_samples": 900, "expand_factor": 0.8})
                        mostrar_resultados(res, f_fn, "x_n+1", ["iter", "x_n", "x_n+1", "error"], [x0 - 1, x0 + 1], plot_cfg)

                        historial = res["historial"]
                        if usar_aitken and len(historial) >= 3:
                            with st.expander("Aceleracion de Aitken (Delta^2)", expanded=False):
                                st.latex(r"\hat{x}_n = x_n - \frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2\,x_{n+1} + x_n}")
                                x_seq = [historial[0]["x_n"]] + [h["x_n+1"] for h in historial]
                                rows = []
                                for k in range(len(x_seq) - 2):
                                    s0, s1, s2 = x_seq[k], x_seq[k + 1], x_seq[k + 2]
                                    d = s2 - 2.0 * s1 + s0
                                    it_equiv = k + 2
                                    try:
                                        resid_orig = abs(float(f_fn(s2)))
                                    except Exception:
                                        resid_orig = None
                                    if abs(d) < 1e-14:
                                        rows.append({"n": k, "x_n": s0, "x_{n+1}": s1, "x_{n+2}": s2,
                                                     "Delta_x_n": s1 - s0, "Delta2_x_n": d,
                                                     "iter_equiv": it_equiv,
                                                     "|f(x_{n+2})|": resid_orig,
                                                     "x_hat_n": None, "|f(x_hat_n)|": None, "error": None})
                                    else:
                                        xh = s0 - (s1 - s0) ** 2 / d
                                        try:
                                            resid_hat = abs(float(f_fn(xh)))
                                        except Exception:
                                            resid_hat = None
                                        rows.append({"n": k, "x_n": s0, "x_{n+1}": s1, "x_{n+2}": s2,
                                                     "Delta_x_n": s1 - s0, "Delta2_x_n": d,
                                                     "iter_equiv": it_equiv,
                                                     "|f(x_{n+2})|": resid_orig,
                                                     "x_hat_n": xh, "|f(x_hat_n)|": resid_hat, "error": abs(xh - s2)})
                                df_a = pd.DataFrame(rows)
                                st.dataframe(_fmt_iter_df(df_a), hide_index=True, use_container_width=True)
                                valid = [r for r in rows if r["x_hat_n"] is not None]
                                if valid:
                                    valid_resid = [r for r in valid if r["|f(x_hat_n)|"] is not None]
                                    best = min(valid_resid, key=lambda r: r["|f(x_hat_n)|"]) if valid_resid else min(valid, key=lambda r: r["error"])
                                    if best["|f(x_hat_n)|"] is not None:
                                        st.caption(f"Mejor Aitken: x_hat ≈ {best['x_hat_n']:.12f}  con |f(x_hat)| ≈ {best['|f(x_hat_n)|']:.4e}")
                                    else:
                                        st.caption(f"Mejor Aitken: x_hat ≈ {best['x_hat_n']:.12f}")

                                    base_conv = next((h["iter"] for h in historial if h["error"] < tol), None)
                                    if base_conv is None:
                                        base_conv = next(
                                            (h["iter"] for h in historial if h.get("|f(x_n+1)|") is not None and h["|f(x_n+1)|"] < tol),
                                            None,
                                        )
                                    aitken_conv = next(
                                        (r["iter_equiv"] for r in valid if r["|f(x_hat_n)|"] is not None and r["|f(x_hat_n)|"] < tol),
                                        None,
                                    )
                                    if base_conv is not None and aitken_conv is not None:
                                        ahorro = base_conv - aitken_conv
                                        st.caption(f"Iteraciones para llegar a tol: Punto Fijo = {base_conv}, Aitken = {aitken_conv} (ahorro = {ahorro}).")
                                    elif aitken_conv is None:
                                        st.info("Aitken no alcanzo la tolerancia con esta g(x).")

                                    df_cmp = pd.DataFrame({
                                        "n": [r["n"] for r in valid],
                                        "original": [r["x_{n+1}"] for r in valid],
                                        "aitken": [r["x_hat_n"] for r in valid],
                                    }).melt("n", var_name="serie", value_name="valor")
                                    st.altair_chart(
                                        alt.Chart(df_cmp).mark_line(point=True)
                                        .encode(x="n:Q", y=alt.Y("valor:Q", title="Aproximacion"),
                                                color="serie:N", tooltip=["n", "serie", "valor"])
                                        .properties(height=280),
                                        use_container_width=True,
                                    )
                    except Exception as e:
                        st.error(str(e))

        with side_col:
            _formulas_panel("Punto Fijo")
