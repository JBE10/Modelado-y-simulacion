import math

import altair as alt
import pandas as pd
import streamlit as st

from biseccion import biseccion
from newton_raphson import newton_raphson
from punto_fijo import punto_fijo
from secante import secante
from diferencias_finitas import diferencias_finitas_funcion, diferencias_finitas_tabla

MATH_NS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "ln": math.log,
    "sqrt": math.sqrt, "abs": abs,
    "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
    "pi": math.pi, "e": math.e,
}


def norm(expr):
    return expr.replace("^", "**").strip()


def make_fn(expr):
    expr = norm(expr)
    def f(x):
        return eval(expr, {"__builtins__": {}}, {"x": x, **MATH_NS})
    return f


def _formulas_panel(algoritmo):
    with st.container(border=True):
        st.markdown("### Formulas")
        if algoritmo == "Biseccion":
            st.latex(r"c_n=\frac{a_n+b_n}{2}")
            st.latex(r"\text{si } f(a_n)\,f(c_n)<0 \Rightarrow [a_{n+1},b_{n+1}]=[a_n,c_n]")
            st.latex(r"\text{si } f(a_n)\,f(c_n)>0 \Rightarrow [a_{n+1},b_{n+1}]=[c_n,b_n]")
            st.caption("Convergencia por intervalo y cambio de signo en [a,b].")
        elif algoritmo == "Punto Fijo":
            st.latex(r"x_{n+1}=g(x_n)")
            st.latex(r"\text{criterio: } |x_{n+1}-x_n|<\mathrm{tol}")
            st.latex(r"\hat{x}_n=x_n-\frac{(x_{n+1}-x_n)^2}{x_{n+2}-2x_{n+1}+x_n}")
            st.caption("Aitken (Delta^2) acelera cuando la sucesion es linealmente convergente.")
        elif algoritmo == "Newton-Raphson":
            st.latex(r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}")
            st.latex(r"\text{criterio: } |x_{n+1}-x_n|<\mathrm{tol}")
            st.caption("Convergencia rapida cerca de la raiz si f'(x) no es casi cero.")
        elif algoritmo == "Secante":
            st.latex(r"x_{n+1}=x_n-f(x_n)\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}")
            st.latex(r"\text{criterio: } |x_{n+1}-x_n|<\mathrm{tol}")
            st.caption("No requiere derivada y suele ser mas rapida que punto fijo.")

        if algoritmo == "Diferencias Finitas":
            st.markdown("**1ra derivada centrada**")
            st.caption("f'(xᵢ) ≈ [f(xᵢ₊₁) - f(xᵢ₋₁)] / (2h)")
            st.markdown("**2da derivada centrada**")
            st.caption("f''(xᵢ) ≈ [f(xᵢ₊₁) - 2f(xᵢ) + f(xᵢ₋₁)] / h²")
            st.caption("🟢 Centrada → O(h²) | 🟡 Extremos → O(h)")
        st.divider()
        st.markdown("### Notas")
        st.caption("Usa `ln(x)` o `log(x)` para logaritmo natural.")
        st.caption("Potencias con `^` o `**`.")


def graficar_fx(f, raiz, x_min, x_max, label="f(x)", n=800, expand=0.8, iter_points=None):
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0

    ancho = abs(x_max - x_min)
    margen = max(ancho * expand, 1.0)
    xa, xb = min(x_min, x_max) - margen, max(x_min, x_max) + margen
    paso = (xb - xa) / max(n - 1, 1)

    datos = []
    segmento = 0
    hubo_gap = False
    for i in range(n):
        xv = xa + i * paso
        try:
            yv = float(f(xv))
        except Exception:
            yv = float("nan")
        if not math.isfinite(yv):
            hubo_gap = True
            continue
        if hubo_gap and datos:
            segmento += 1
            hubo_gap = False
        datos.append({"x": xv, "y": yv, "segmento": segmento})

    if not datos:
        st.warning("No se pudo graficar f(x) en el rango actual (dominio invalido o valores no finitos).")
        return

    df = pd.DataFrame(datos)
    yvals = df["y"]
    q_low = float(yvals.quantile(0.02))
    q_high = float(yvals.quantile(0.98))
    if q_high <= q_low:
        q_low, q_high = float(yvals.min()), float(yvals.max())
    ypad = max((q_high - q_low) * 0.15, 0.5)

    curva = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5, color="#2563eb")
        .encode(
            x=alt.X("x:Q", title="x", scale=alt.Scale(domain=[xa, xb])),
            y=alt.Y("y:Q", title=label, scale=alt.Scale(domain=[q_low - ypad, q_high + ypad])),
            detail="segmento:N",
            tooltip=[alt.Tooltip("x:Q", format=".8f"), alt.Tooltip("y:Q", format=".8e")],
        )
        .properties(height=400, title="Comportamiento de f(x)")
    )
    eje_x = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#6b7280", strokeDash=[4, 4]).encode(y="y:Q")
    eje_y = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#6b7280", strokeDash=[4, 4]).encode(x="x:Q")

    capas = [curva, eje_x, eje_y]
    try:
        fr = float(f(raiz))
    except Exception:
        fr = float("nan")
    if math.isfinite(fr):
        df_r = pd.DataFrame({"x": [raiz], "y": [fr], "label": [f"raiz ≈ {raiz:.8f}"]})
        punto = alt.Chart(df_r).mark_point(size=180, color="#ef4444", filled=True, stroke="#fff", strokeWidth=1.5).encode(x="x:Q", y="y:Q")
        etiq = alt.Chart(df_r).mark_text(align="left", dx=10, dy=-10, fontSize=12, fontWeight="bold", color="#ef4444").encode(x="x:Q", y="y:Q", text="label:N")
        capas.extend([punto, etiq])

    if iter_points:
        it_datos = []
        for idx, xv in enumerate(iter_points, start=1):
            try:
                yv = float(f(float(xv)))
                if math.isfinite(yv):
                    it_datos.append({"iter": idx, "x": float(xv), "y": yv})
            except Exception:
                continue
        if it_datos:
            df_i = pd.DataFrame(it_datos)
            capas.append(
                alt.Chart(df_i)
                .mark_point(size=70, color="#16a34a", filled=True)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    tooltip=[alt.Tooltip("iter:Q"), alt.Tooltip("x:Q", format=".8f"), alt.Tooltip("y:Q", format=".8e")],
                )
            )

    chart = alt.layer(*capas).interactive().configure_axis(grid=True, gridColor="#e5e7eb", gridOpacity=0.35)
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Rango mostrado: [{xa:.4f}, {xb:.4f}] con {n} muestras.")


def mostrar_resultados(res, f_fn, y_col, tooltip_cols, x_range, plot_cfg):
    raiz = res["raiz"]
    with st.container(border=True):
        if res["convergio"]:
            st.success(res["justificacion"])
        else:
            st.warning(res["justificacion"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Raiz aprox.", f"{raiz:.12f}")
        c2.metric("Iteraciones", res["iteraciones"])
        try:
            c3.metric("|f(raiz)|", f"{abs(f_fn(raiz)):.4e}")
        except Exception:
            c3.metric("|f(raiz)|", "N/A")

    df_h = pd.DataFrame(res["historial"])
    if df_h.empty:
        return

    with st.container(border=True):
        st.markdown("### Tabla de iteraciones")
        st.dataframe(df_h, hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown("### Graficas de convergencia")
        g1, g2 = st.columns(2)
        with g1:
            st.altair_chart(
                alt.Chart(df_h).mark_line(point=True)
                .encode(x="iter:Q", y=alt.Y(f"{y_col}:Q", title="x"), tooltip=tooltip_cols)
                .properties(height=280),
                use_container_width=True,
            )
        with g2:
            st.altair_chart(
                alt.Chart(df_h).mark_line(point=True, color="#d62728")
                .encode(x="iter:Q", y=alt.Y("error:Q"), tooltip=["iter", "error"])
                .properties(height=280),
                use_container_width=True,
            )

    with st.container(border=True):
        st.markdown("### Grafica de f(x)")
        iter_points = df_h[y_col].dropna().tolist() if y_col in df_h.columns else None
        graficar_fx(
            f_fn,
            raiz,
            x_range[0],
            x_range[1],
            n=plot_cfg["n_samples"],
            expand=plot_cfg["expand_factor"],
            iter_points=iter_points,
        )


# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Metodos Numericos", layout="wide")
st.title("Metodos Numericos")
st.caption("Funciones: sin, cos, tan, exp, log, ln, sqrt, cbrt, abs — Constantes: pi, e — Potencias: ^ o **")
st.divider()

with st.sidebar:
    st.header("Configuracion")
    algoritmo = st.selectbox("Algoritmo", ["Biseccion", "Punto Fijo", "Newton-Raphson", "Secante", "Diferencias Finitas"])
    st.divider()
    st.markdown("### Vista de f(x)")
    n_samples = st.slider("Muestras", min_value=200, max_value=2000, step=100, value=900)
    expand_factor = st.slider("Factor de expansion del rango", min_value=0.2, max_value=2.0, step=0.1, value=0.8)
    st.caption("Aumenta muestras para curvas mas detalladas.")

plot_cfg = {"n_samples": int(n_samples), "expand_factor": float(expand_factor)}


# ── BISECCION ──────────────────────────────────────────────────────────────
if algoritmo == "Biseccion":
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
                mostrar_resultados(res, f_fn, "c", ["iter", "c", "f(c)", "error"], [a, b], plot_cfg)
            except Exception as e:
                st.error(str(e))
    with side_col:
        _formulas_panel("Biseccion")


# ── PUNTO FIJO ─────────────────────────────────────────────────────────────
if algoritmo == "Punto Fijo":
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
                            st.dataframe(df_a, hide_index=True, use_container_width=True)
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


# ── NEWTON-RAPHSON ─────────────────────────────────────────────────────────
if algoritmo == "Newton-Raphson":
    main_col, side_col = st.columns([2.3, 1.0], gap="large")
    with main_col:
        with st.container(border=True):
            st.subheader("Newton-Raphson")
            st.caption("Metodo abierto con derivada explicita.")
            with st.form("newton"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f = st.text_input("f(x)", "x**3 - x - 2", key="f_nr")
                    expr_df = st.text_input("f'(x)", "3*x**2 - 1", key="df_nr")
                    x0 = st.number_input("x0", value=1.5, format="%.10f", key="x0_nr")
                with c2:
                    tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_nr")
                    mi = st.number_input("Max iter", value=50, min_value=1, step=1, key="mi_nr")
                run = st.form_submit_button("Calcular", type="primary")

        if run:
            if not expr_f.strip() or not expr_df.strip():
                st.error("Debes ingresar f(x) y f'(x).")
            else:
                try:
                    f_fn = make_fn(expr_f)
                    df_fn = make_fn(expr_df)
                    res = newton_raphson(f_fn, df_fn, x0, tol=tol, max_iter=int(mi))
                    mostrar_resultados(
                        res,
                        f_fn,
                        "x_n+1",
                        ["iter", "x_n", "f(x_n)", "f'(x_n)", "x_n+1", "error"],
                        [min(x0, res["raiz"]) - 1, max(x0, res["raiz"]) + 1],
                        plot_cfg,
                    )
                except Exception as e:
                    st.error(str(e))
    with side_col:
        _formulas_panel("Newton-Raphson")


# ── SECANTE ────────────────────────────────────────────────────────────────
if algoritmo == "Secante":
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
                    mostrar_resultados(
                        res,
                        f_fn,
                        "x_{n+1}",
                        ["iter", "x_n", "f(x_n)", "x_{n+1}", "error"],
                        [min(x0, x1, res["raiz"]) - 1, max(x0, x1, res["raiz"]) + 1],
                        plot_cfg,
                    )
                except Exception as e:
                    st.error(str(e))
    with side_col:
        _formulas_panel("Secante")


# ── DIFERENCIAS FINITAS ────────────────────────────────────────────────────
if algoritmo == "Diferencias Finitas":
    main_col, side_col = st.columns([2.3, 1.0], gap="large")
    with main_col:
        # ── Panel de fórmulas completo (dentro del área principal) ────────
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

        modo = st.radio(
            "Modo",
            ["Función continua (ingreso f(x) y h)", "Tabla de datos discretos"],
            horizontal=True,
        )

        # ── MODO FUNCIÓN ────────────────────────────────────────────────
        if modo == "Función continua (ingreso f(x) y h)":
            with st.container(border=True):
                st.subheader("Diferencias Finitas — Función continua")
                st.caption("Estima f\' y f\'\' en un punto x₀ con paso h usando diferencias centradas.")
                with st.form("dif_fin_fn"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f  = st.text_input("f(x)", "x**3 - 2*x + 1", key="df_f")
                        expr_d1 = st.text_input("f\' exacta (opcional)", "3*x**2 - 2", key="df_d1")
                        expr_d2 = st.text_input("f\'\' exacta (opcional)", "6*x", key="df_d2")
                    with c2:
                        x0_val  = st.number_input("x₀", value=2.0, format="%.6f", key="df_x0")
                        h_val   = st.number_input("h (paso)", value=0.1, min_value=1e-10,
                                                  format="%.8f", key="df_h")
                        n_h     = st.slider("Comparar N valores de h", 1, 20, 8, key="df_nh")
                    run_fn = st.form_submit_button("Calcular", type="primary")

            if run_fn:
                try:
                    f_fn   = make_fn(expr_f)
                    d1_fn  = make_fn(expr_d1) if expr_d1.strip() else None
                    d2_fn  = make_fn(expr_d2) if expr_d2.strip() else None
                    exacta_d1 = float(d1_fn(x0_val)) if d1_fn else None
                    exacta_d2 = float(d2_fn(x0_val)) if d2_fn else None

                    res = diferencias_finitas_funcion(
                        f_fn, x0_val, h_val,
                        exacta_d1=exacta_d1,
                        exacta_d2=exacta_d2,
                    )

                    # Métricas principales
                    with st.container(border=True):
                        st.markdown("### Resultado en x₀")
                        cols = st.columns(4)
                        cols[0].metric("f(x₀)",   f"{res['f(x0)']:.10f}")
                        cols[1].metric("f'(x₀) centrada",  f"{res['d1']:.10f}")
                        cols[2].metric("f''(x₀) centrada", f"{res['d2']:.10f}")
                        cols[3].metric("h usado", f"{h_val:.2e}")

                        if exacta_d1 is not None:
                            st.caption(
                                f"Valor exacto f'(x₀) = {exacta_d1:.10f}  →  "
                                f"Error abs = {res['error_d1']:.4e}"
                            )
                        if exacta_d2 is not None:
                            st.caption(
                                f"Valor exacto f''(x₀) = {exacta_d2:.10f}  →  "
                                f"Error abs = {res['error_d2']:.4e}"
                            )

                        st.latex(r"f'(x_0) \approx \frac{f(x_0+h)\,-\,f(x_0-h)}{2h}")
                        st.latex(r"f''(x_0) \approx \frac{f(x_0+h)\,-\,2f(x_0)\,+\,f(x_0-h)}{h^2}")

                    # Tabla de evaluaciones
                    with st.container(border=True):
                        st.markdown("### Evaluaciones en x₀")
                        df_eval = pd.DataFrame([{
                            "Punto":   "x₀ - h",
                            "x":       x0_val - h_val,
                            "f(x)":    res["f(x0-h)"],
                        }, {
                            "Punto":   "x₀",
                            "x":       x0_val,
                            "f(x)":    res["f(x0)"],
                        }, {
                            "Punto":   "x₀ + h",
                            "x":       x0_val + h_val,
                            "f(x)":    res["f(x0+h)"],
                        }])
                        st.dataframe(df_eval, hide_index=True, use_container_width=True)

                    # Análisis de convergencia: error vs h
                    if exacta_d1 is not None or exacta_d2 is not None:
                        with st.container(border=True):
                            st.markdown("### Convergencia del error según h")
                            hs = [h_val * (0.5 ** k) for k in range(n_h)]
                            rows_err = []
                            for hk in hs:
                                rk = diferencias_finitas_funcion(
                                    f_fn, x0_val, hk,
                                    exacta_d1=exacta_d1,
                                    exacta_d2=exacta_d2,
                                )
                                rows_err.append({
                                    "h":         hk,
                                    "f'(x₀)": rk["d1"],
                                    "f''(x₀)": rk["d2"],
                                    "error f'": rk.get("error_d1"),
                                    "error f''": rk.get("error_d2"),
                                })
                            df_err = pd.DataFrame(rows_err)
                            st.dataframe(
                                df_err.style.format({
                                    "h": "{:.2e}",
                                    "f'(x₀)": "{:.10f}",
                                    "f''(x₀)": "{:.10f}",
                                    "error f'": "{:.4e}",
                                    "error f''": "{:.4e}",
                                }),
                                hide_index=True,
                                use_container_width=True,
                            )

                            # Gráfica error vs h
                            err_cols = []
                            if exacta_d1 is not None:
                                err_cols.append("error f'")
                            if exacta_d2 is not None:
                                err_cols.append("error f''")

                            df_melt = df_err[["h"] + err_cols].melt("h", var_name="derivada", value_name="error")
                            chart_err = (
                                alt.Chart(df_melt)
                                .mark_line(point=True)
                                .encode(
                                    x=alt.X("h:Q", title="h", scale=alt.Scale(type="log")),
                                    y=alt.Y("error:Q", title="Error absoluto", scale=alt.Scale(type="log")),
                                    color="derivada:N",
                                    tooltip=["h", "derivada", "error"],
                                )
                                .properties(height=300, title="Error vs h (escala log-log)")
                            )
                            st.altair_chart(chart_err, use_container_width=True)
                            st.caption("La pendiente ~2 en log-log confirma el orden O(h²) de las diferencias centradas.")

                    # Gráfica f(x) con los tres puntos usados
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
                        curva_g = (
                            alt.Chart(df_g)
                            .mark_line(color="#3b82f6", strokeWidth=2)
                            .encode(x="x:Q", y="y:Q", tooltip=["x", "y"])
                        )
                        pts_data = pd.DataFrame([
                            {"x": x0_val - h_val, "y": res["f(x0-h)"], "label": "x₀-h"},
                            {"x": x0_val,         "y": res["f(x0)"],   "label": "x₀"},
                            {"x": x0_val + h_val, "y": res["f(x0+h)"], "label": "x₀+h"},
                        ])
                        puntos_g = (
                            alt.Chart(pts_data)
                            .mark_point(size=150, filled=True)
                            .encode(
                                x="x:Q", y="y:Q",
                                color=alt.Color("label:N", legend=alt.Legend(title="Punto")),
                                tooltip=["label", "x", "y"],
                            )
                        )
                        st.altair_chart(
                            (curva_g + puntos_g)
                            .properties(height=320, title="f(x) con puntos de diferenciación")
                            .interactive(),
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(str(e))

        # ── MODO TABLA ────────────────────────────────────────────────────
        else:
            with st.container(border=True):
                st.subheader("Diferencias Finitas — Tabla de datos discretos")
                st.caption(
                    "Ingresá los puntos como pares x,y separados por punto y coma. "
                    "Interior → centrada | Extremos → progresiva/regresiva."
                )
                with st.form("dif_fin_tabla"):
                    puntos_raw = st.text_area(
                        "Puntos (x, y) — uno por línea o separados por ;",
                        value="0, 1\n1, 2\n2, 0\n3, 2\n4, 3",
                        height=160,
                        key="df_tabla",
                    )
                    run_tab = st.form_submit_button("Calcular", type="primary")

            if run_tab:
                try:
                    lineas = [l.strip() for l in puntos_raw.replace(";", "\n").splitlines() if l.strip()]
                    xs_t, ys_t = [], []
                    for linea in lineas:
                        partes = linea.split(",")
                        if len(partes) != 2:
                            raise ValueError(f"Formato incorrecto en línea: '{linea}'")
                        xs_t.append(float(partes[0]))
                        ys_t.append(float(partes[1]))

                    res_tab = diferencias_finitas_tabla(xs_t, ys_t)

                    with st.container(border=True):
                        st.markdown("### Tabla de derivadas")
                        df_tab = pd.DataFrame(res_tab["filas"])
                        df_tab = df_tab.rename(columns={
                            "i": "i", "x_i": "x_i", "y_i": "y_i (f)",
                            "f'(x_i)": "f'(x_i)", "tipo_d1": "Tipo 1ra der",
                            "f''(x_i)": "f''(x_i)", "tipo_d2": "Tipo 2da der",
                        })
                        st.dataframe(
                            df_tab.style.applymap(
                                lambda v: "color: #86efac; font-weight:600"
                                if v == "Centrada" else
                                ("color: #fbbf24" if v in ("Progresiva", "Regresiva") else ""),
                                subset=["Tipo 1ra der", "Tipo 2da der"],
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.caption("🟢 Centrada (O(h²))  🟡 Progresiva/Regresiva (O(h))")

                    with st.container(border=True):
                        st.markdown("### Gráficas")
                        df_plot = pd.DataFrame(res_tab["filas"])
                        g1, g2 = st.columns(2)
                        with g1:
                            st.altair_chart(
                                alt.Chart(df_plot)
                                .mark_line(point=True, color="#3b82f6")
                                .encode(
                                    x=alt.X("x_i:Q", title="x"),
                                    y=alt.Y("y_i:Q", title="f(x)"),
                                    tooltip=["x_i", "y_i"],
                                )
                                .properties(height=270, title="Datos f(x)"),
                                use_container_width=True,
                            )
                        with g2:
                            df_d1 = df_plot[["x_i", "f'(x_i)", "tipo_d1"]].copy()
                            st.altair_chart(
                                alt.Chart(df_d1)
                                .mark_line(point=True)
                                .encode(
                                    x=alt.X("x_i:Q", title="x"),
                                    y=alt.Y("f'(x_i):Q", title="f'(x)"),
                                    color=alt.Color("tipo_d1:N", title="Esquema"),
                                    tooltip=["x_i", "f'(x_i)", "tipo_d1"],
                                )
                                .properties(height=270, title="Primera derivada estimada"),
                                use_container_width=True,
                            )

                except Exception as e:
                    st.error(str(e))

    with side_col:
        _formulas_panel("Diferencias Finitas")
