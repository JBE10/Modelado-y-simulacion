"""steffensen_module.py — Steffensen-Aitken UI Module."""
import math
import altair as alt
import pandas as pd
import streamlit as st
from ui_modules.base import DashboardModule
from ui_utils import make_fn, graficar_fx, _fmt_iter_df
from steffensen import steffensen
from punto_fijo import punto_fijo as _pf


class SteffensenModule(DashboardModule):
    @property
    def name(self) -> str:
        return "Steffensen-Aitken"

    def render(self):
        main_col, side_col = st.columns([2.3, 1.0], gap="large")
        with main_col:
            with st.container(border=True):
                st.subheader("Steffensen-Aitken (Δ²)")
                st.caption(
                    "Método de punto fijo **acelerado**: cada iteración usa dos evaluaciones "
                    "de g para producir x̂_n = x_n − (g(x_n)−x_n)² / (g(g(x_n))−2g(x_n)+x_n) "
                    "y ese x̂ se convierte en el siguiente iterado (no es post-proceso)."
                )
                with st.form("steffensen"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f_st = st.text_input("f(x)", "2*x*cos(x) - (x-2)**2", key="f_st")
                        expr_g_st = st.text_input("g(x)  — función de iteración", "2 - sqrt(2*x*cos(x))", key="g_st")
                        x0_st = st.number_input("x₀ (semilla)", value=1.0, format="%.10f", key="x0_st")
                    with c2:
                        tol_st = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.2e", key="tol_st")
                        mi_st = st.number_input("Máx. iteraciones Steffensen", value=50, min_value=1, step=1, key="mi_st")
                    run_st = st.form_submit_button("Calcular", type="primary")

            if run_st:
                if not expr_f_st.strip() or not expr_g_st.strip():
                    st.error("Ingresá f(x) y g(x).")
                else:
                    try:
                        f_fn_st = make_fn(expr_f_st)
                        g_fn_st = make_fn(expr_g_st)
                        res_st = steffensen(g_fn_st, x0_st, tol=tol_st, max_iter=int(mi_st), f=f_fn_st)
                        raiz_st = res_st["raiz"]
                        plot_cfg = st.session_state.get("plot_cfg", {"n_samples": 900, "expand_factor": 0.8})

                        with st.container(border=True):
                            if res_st["convergio"]:
                                st.success(res_st["justificacion"])
                            else:
                                st.warning(res_st["justificacion"])
                            c1m, c2m, c3m = st.columns(3)
                            c1m.metric("Raíz aproximada", f"{raiz_st:.10f}")
                            c2m.metric("Iteraciones Steffensen", res_st["iteraciones"])
                            try:
                                c3m.metric("|f(raíz)|", f"{abs(f_fn_st(raiz_st)):.4e}")
                            except Exception:
                                c3m.metric("|f(raíz)|", "N/A")

                        df_st = pd.DataFrame(res_st["historial"])
                        if not df_st.empty:
                            with st.container(border=True):
                                st.markdown("### Tabla de iteraciones Steffensen")
                                st.caption(
                                    "Cada fila = 1 paso completo (2 evaluaciones de g). "
                                    "**x_hat** es el nuevo iterado, no un post-proceso."
                                )
                                st.dataframe(_fmt_iter_df(df_st), hide_index=True, use_container_width=True)

                            with st.container(border=True):
                                st.markdown("### Convergencia del error")
                                g1, g2 = st.columns(2)
                                with g1:
                                    st.altair_chart(
                                        alt.Chart(df_st).mark_line(point=True, color="#7c3aed")
                                        .encode(
                                            x=alt.X("iter:Q", title="Iteración Steffensen"),
                                            y=alt.Y("x_hat:Q", title="x̂_n"),
                                            tooltip=["iter", "x_hat", "error"],
                                        ).properties(height=280, title="x̂_n por iteración"),
                                        use_container_width=True,
                                    )
                                with g2:
                                    df_err_st = df_st[df_st["error"] > 0].copy()
                                    if not df_err_st.empty:
                                        df_err_st["log10_error"] = df_err_st["error"].apply(
                                            lambda e: math.log10(e) if e > 0 else None
                                        )
                                        st.altair_chart(
                                            alt.Chart(df_err_st).mark_line(point=True, color="#ef4444")
                                            .encode(
                                                x=alt.X("iter:Q", title="Iteración"),
                                                y=alt.Y("log10_error:Q", title="log₁₀(|error|)"),
                                                tooltip=["iter", "log10_error", "error"],
                                            ).properties(height=280, title="Convergencia logarítmica"),
                                            use_container_width=True,
                                        )

                            errores = [r["error"] for r in res_st["historial"] if r["error"] > 0]
                            if len(errores) >= 3:
                                with st.container(border=True):
                                    st.markdown("### Orden de convergencia estimado")
                                    p_vals = []
                                    for k_idx in range(2, len(errores)):
                                        e0, e1, e2 = errores[k_idx - 2], errores[k_idx - 1], errores[k_idx]
                                        if e1 > 0 and e0 > 0:
                                            denom = math.log(abs(e1 / e0))
                                            if abs(denom) > 1e-15:
                                                p = math.log(abs(e2 / e1)) / denom
                                                if math.isfinite(p) and 0 < p < 10:
                                                    p_vals.append(p)
                                    if p_vals:
                                        p_est = sum(p_vals[-3:]) / len(p_vals[-3:])
                                        st.latex(
                                            rf"p \approx \frac{{\ln|e_{{n+1}}/e_n|}}{{\ln|e_n/e_{{n-1}}|}} "
                                            rf"\approx {p_est:.4f}"
                                        )
                                        if p_est < 1.3:
                                            st.caption("⚡ Convergencia **lineal** (p ≈ 1).")
                                        elif p_est < 1.8:
                                            st.caption("⚡ Convergencia **superlineal** (p ≈ 1.62). Típico de Steffensen-Aitken.")
                                        else:
                                            st.caption("⚡ Convergencia **cuadrática** (p ≈ 2).")

                            with st.container(border=True):
                                st.markdown("### Gráfica de f(x)")
                                iter_pts = [r["x_hat"] for r in res_st["historial"]]
                                x_rng = [min(x0_st, raiz_st) - 1, max(x0_st, raiz_st) + 1]
                                graficar_fx(f_fn_st, raiz_st, x_rng[0], x_rng[1],
                                            n=plot_cfg["n_samples"],
                                            expand=plot_cfg["expand_factor"],
                                            iter_points=iter_pts)

                            with st.container(border=True):
                                st.markdown("### ⚖️ Comparación: Steffensen-Aitken vs Punto Fijo puro")
                                try:
                                    res_pf = _pf(g_fn_st, x0_st, tol=tol_st, max_iter=200, f=f_fn_st)
                                    cols_cmp = st.columns(2)
                                    cols_cmp[0].metric(
                                        "🟣 Steffensen-Aitken",
                                        f"{res_st['iteraciones']} iteraciones",
                                        f"Raíz = {raiz_st:.8f}",
                                    )
                                    cols_cmp[1].metric(
                                        "🔵 Punto Fijo puro",
                                        f"{res_pf['iteraciones']} iteraciones",
                                        f"Raíz = {res_pf['raiz']:.8f}",
                                    )
                                    ahorro = res_pf["iteraciones"] - res_st["iteraciones"]
                                    if ahorro > 0:
                                        st.success(
                                            f"Steffensen-Aitken convergió en **{res_st['iteraciones']}** pasos "
                                            f"vs **{res_pf['iteraciones']}** de Punto Fijo — "
                                            f"ahorro de **{ahorro}** iteraciones ({ahorro/res_pf['iteraciones']*100:.0f}%)."
                                        )
                                    else:
                                        st.info("Ambos métodos convergen en un número similar de iteraciones.")
                                except Exception:
                                    pass

                    except Exception as e:
                        st.error(str(e))

        with side_col:
            with st.container(border=True):
                st.markdown("### Fórmula Δ²")
                st.latex(r"x' = g(x_n)")
                st.latex(r"x'' = g(x')")
                st.latex(r"\hat{x}_n = x_n - \frac{(x' - x_n)^2}{x'' - 2x' + x_n}")
                st.caption("x̂_n → siguiente iterado (no post-proceso)")
                st.divider()
                st.markdown("### Diferencia clave")
                st.info("**Punto Fijo + Aitken**: itera g normalmente, muestra tabla Aitken al final.")
                st.success("**Steffensen-Aitken**: usa x̂_n como nuevo x_n en cada paso → convergencia superlineal.")
                st.divider()
                st.markdown("### Condición de Lipschitz")
                st.latex(r"|g'(x^*)| < 1 \text{ (local)}")
                st.caption("Steffensen converge incluso cuando esta condición es marginal en todo el intervalo.")
                st.divider()
                st.markdown("### Evaluaciones de g por iter.")
                st.latex(r"2 \times \text{iter\_Steffensen} \approx \text{iter\_PF}")
                st.caption("Usa el doble de evaluaciones de g por paso, pero muchos menos pasos.")
