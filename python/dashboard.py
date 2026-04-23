import math

import altair as alt
import pandas as pd
import streamlit as st

from biseccion import biseccion
from newton_raphson import newton_raphson
from punto_fijo import punto_fijo
from secante import secante
from diferencias_finitas import (
    diferencias_finitas_funcion,
    diferencias_finitas_tabla,
    interpolacion_newton_divididas,
)
from integracion_numerica import (
    integracion_rectangulo,
    integracion_trapecio,
    integracion_simpson_13,
    integracion_simpson_38,
    integral_referencia,
)
from montecarlo import (
    estimar_pi,
    integracion_1d_mc,
    integracion_2d_mc,
    multi_run_1d,
    simular_monty_hall,
)

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
    import sympy
    expr = norm(expr).replace("ln(", "log(")
    MATH_SYMPY_NS = {
        "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
        "sqrt": sympy.sqrt, "abs": sympy.Abs,
        "cbrt": lambda z: sympy.root(z, 3),
    }
    try:
        parsed_expr = sympy.sympify(expr, locals=MATH_SYMPY_NS)
        x_sym = sympy.Symbol('x')
        f_lambdified = sympy.lambdify(x_sym, parsed_expr, "math")
        def f(x):
            try:
                # Retornamos float puro
                val = f_lambdified(x)
                if isinstance(val, complex) and val.imag == 0:
                    val = val.real
                v = float(val)
                if not math.isfinite(v):
                    raise ValueError
                return v
            except Exception:
                try:
                    lim = sympy.limit(parsed_expr, x_sym, x)
                    if lim.is_real and lim.is_finite:
                        return float(lim.evalf())
                except Exception:
                    pass
                return float('nan')
        return f
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida: {e}")


def parse_number_cell(s: str) -> float:
    """Convierte una celda a float de forma segura usando sympy."""
    import sympy
    t = s.strip()
    if not t:
        raise ValueError("Celda vacía.")
    try:
        return float(t)
    except ValueError:
        pass
        
    MATH_SYMPY_NS = {
        "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
        "sqrt": sympy.sqrt, "abs": sympy.Abs,
        "cbrt": lambda z: sympy.root(z, 3),
    }
    expr = norm(t).replace("ln(", "log(")
    try:
        parsed = sympy.sympify(expr, locals=MATH_SYMPY_NS)
        fv = float(parsed.evalf())
        if not math.isfinite(fv):
            raise ValueError(f"Valor no finito: {s!r}")
        return fv
    except Exception as e:
        raise ValueError(f"Expresión numérica inválida '{s}': {e}")


def make_fn_2d(expr):
    import sympy
    expr = norm(expr).replace("ln(", "log(")
    MATH_SYMPY_NS = {
        "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
        "sqrt": sympy.sqrt, "abs": sympy.Abs,
        "cbrt": lambda z: sympy.root(z, 3),
    }
    try:
        parsed_expr = sympy.sympify(expr, locals=MATH_SYMPY_NS)
        x_sym, y_sym = sympy.Symbol('x'), sympy.Symbol('y')
        f_lambdified = sympy.lambdify((x_sym, y_sym), parsed_expr, "math")
        def f(xv, yv):
            try:
                val = f_lambdified(xv, yv)
                if isinstance(val, complex) and val.imag == 0:
                    val = val.real
                return float(val)
            except Exception:
                return float('nan')
        return f
    except Exception as e:
        raise ValueError(f"Expresión matemática 2D inválida: {e}")


def expr_integrando_latex(expr_py: str) -> str:
    """Convierte la expresión Python de f(x) a LaTeX (integrando)."""
    try:
        from sympy import (
            Abs,
            E,
            cos,
            exp,
            latex,
            log,
            pi,
            root,
            sin,
            sqrt,
            sympify,
            tan,
        )

        s = norm(expr_py).replace("ln(", "log(")
        loc = {
            "pi": pi,
            "E": E,
            "e": E,
            "exp": exp,
            "log": log,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "sqrt": sqrt,
            "abs": Abs,
            "cbrt": lambda z: root(z, 3),
        }
        return latex(sympify(s, locals=loc))
    except Exception:
        t = (
            expr_py.strip()
            .replace("\\", r"\textbackslash ")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("&", r"\&")
        )
        return rf"\text{{{t}}}"


def _latex_float(x: float, ndec: int) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return r"\text{—}"
    return f"{float(x):.{ndec}f}"


def _latex_bound(x: float, ndec: int) -> str:
    """Límite a y b en notación LaTeX (enteros sin cola decimal)."""
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "?"
    xf = float(x)
    if abs(xf - round(xf)) < 1e-14 * max(1.0, abs(xf)):
        return str(int(round(xf)))
    return f"{xf:.{ndec}f}"


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
            st.markdown("**Interpolación (Newton)**")
            st.latex(
                r"P(x)=\sum_{k=0}^{n}\frac{\Delta^k y_0}{k!}\,s(s-1)\cdots(s-k+1),"
                r"\quad s=\frac{x-x_0}{h}"
            )
        elif algoritmo == "Integración Numérica":
            st.markdown("**Regla del Rectángulo (Medio)**")
            st.latex(r"I_r = h \sum f(x_{k+1/2}) \quad \to \mathcal{O}(h^2)")
            st.latex(r"E_t = -\frac{(b-a)h^2}{24} f''(\xi)")
            st.markdown("**Regla del Trapecio**")
            st.latex(r"I_t = \frac{h}{2} \left[ f(x_0) + 2\sum f(x_k) + f(x_n) \right] \to \mathcal{O}(h^2)")
            st.latex(r"E_t = -\frac{(b-a)h^2}{12} f''(\xi)")
            st.markdown("**Regla de Simpson 1/3 (n par)**")
            st.latex(r"I_s = \frac{h}{3} \left[ f(0) + 4\sum_{impares}f(x) + 2\sum_{pares}f(x) + f(n) \right]")
            st.latex(r"E_t = -\frac{(b-a)h^4}{180} f^{(4)}(\xi)")
            st.markdown("**Regla de Simpson 3/8**")
            st.latex(r"I_{s38} = \frac{3h}{8} \left[ f(0) + 3\sum f(x) + 2\sum_{3k}f(x) + f(n) \right]")
            st.latex(r"E_t = -\frac{(b-a)h^4}{80} f^{(4)}(\xi)")
            st.caption(
                "En el panel principal: integral de referencia (quad) y error de truncamiento "
                r"$E_{\mathrm{trunc}} \approx I_{\mathrm{ref}} - I_{\mathrm{aprox}}$ (sin redondeo de máquina)."
            )
        elif algoritmo == "Monte Carlo":
            st.markdown("**Estimación de π**")
            st.latex(r"\pi \approx 4 \times \frac{\text{Aciertos}}{N}")
            st.markdown("**Integración 1D**")
            st.latex(r"\hat{I} = (b-a) \frac{1}{N} \sum_{i=1}^{N} f(x_i)")
            st.markdown("**Intervalo de Confianza (IC)**")
            st.latex(r"IC = \hat{I} \pm Z_{\alpha/2} \frac{\sigma}{\sqrt{N}} (b-a)")
            st.caption("Donde Z = 1.96 para 95% de confianza estadística.")
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
    st.caption(f"Rango mostrado: [{xa:.8f}, {xb:.8f}] con {n} muestras.")


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

    # ── Convergencia logarítmica y orden p ──
    if "error" in df_h.columns and len(df_h) >= 3:
        errores = df_h["error"].dropna().tolist()
        errores_positivos = [e for e in errores if e > 0]
        if len(errores_positivos) >= 3:
            with st.container(border=True):
                st.markdown("### Convergencia Logarítmica")
                import numpy as np
                df_log = pd.DataFrame({
                    "iter": list(range(1, len(errores_positivos) + 1)),
                    "log₁₀(error)": [math.log10(e) for e in errores_positivos]
                })
                chart_log = alt.Chart(df_log).mark_line(point=True, color="#7c3aed").encode(
                    x=alt.X("iter:Q", title="Iteración"),
                    y=alt.Y("log₁₀(error):Q", title="log₁₀(|error|)"),
                    tooltip=["iter", "log₁₀(error)"]
                ).properties(height=280, title="log₁₀(error) vs Iteración")
                st.altair_chart(chart_log, use_container_width=True)

                # Orden de convergencia estimado
                p_vals = []
                for k in range(2, len(errores_positivos)):
                    e_k   = errores_positivos[k]
                    e_k1  = errores_positivos[k-1]
                    e_k2  = errores_positivos[k-2]
                    if e_k1 > 0 and e_k2 > 0 and e_k > 0:
                        denom = math.log(abs(e_k1 / e_k2))
                        if abs(denom) > 1e-15:
                            p = math.log(abs(e_k / e_k1)) / denom
                            if math.isfinite(p) and 0 < p < 10:
                                p_vals.append(p)
                if p_vals:
                    p_est = sum(p_vals[-3:]) / len(p_vals[-3:])  # promedio de los últimos vals
                    st.latex(rf"p \approx \frac{{\ln|e_{{n+1}}/e_n|}}{{\ln|e_n/e_{{n-1}}|}} \approx {p_est:.4f}")
                    if p_est < 1.3:
                        st.caption("⚡ Convergencia **lineal** (p ≈ 1). Típico de Bisección y Punto Fijo.")
                    elif p_est < 1.8:
                        st.caption("⚡ Convergencia **superlineal** (p ≈ 1.62). Típico de la Secante.")
                    else:
                        st.caption("⚡ Convergencia **cuadrática** (p ≈ 2). Típico de Newton-Raphson.")

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
    algoritmo = st.selectbox("Algoritmo", ["Biseccion", "Punto Fijo", "Newton-Raphson", "Secante", "Comparar Raíces", "Diferencias Finitas", "Integración Numérica", "Monte Carlo"])
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
                        x_sym = sympy.Symbol('x')
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


# ── COMPARAR RAÍCES ───────────────────────────────────────────────────────
if algoritmo == "Comparar Raíces":
    main_col, side_col = st.columns([2.3, 1.0], gap="large")
    with main_col:
        with st.container(border=True):
            st.subheader("Comparador de Métodos de Raíces")
            st.caption("Ejecuta Bisección, Newton-Raphson y Secante sobre la misma f(x) y compara convergencia.")
            with st.form("comparar_raices"):
                c1, c2 = st.columns(2)
                with c1:
                    expr_f = st.text_input("f(x)", "x**3 - x - 2", key="cmp_f")
                    expr_df = st.text_input("f'(x) (para Newton, deja vacío para auto)", "", key="cmp_df")
                    a_cmp = st.number_input("a (para Bisección)", value=1.0, format="%.10f", key="cmp_a")
                    b_cmp = st.number_input("b (para Bisección)", value=2.0, format="%.10f", key="cmp_b")
                with c2:
                    x0_cmp = st.number_input("x0 (para Newton/Secante)", value=1.0, format="%.10f", key="cmp_x0")
                    x1_cmp = st.number_input("x1 (para Secante)", value=2.0, format="%.10f", key="cmp_x1")
                    tol_cmp = st.number_input("Tolerancia", value=1e-10, min_value=1e-15, format="%.15f", key="cmp_tol")
                    mi_cmp = st.number_input("Max iter", value=100, min_value=1, step=1, key="cmp_mi")
                run_cmp = st.form_submit_button("Comparar", type="primary")

        if run_cmp and expr_f.strip():
            try:
                f_fn = make_fn(expr_f)
                resultados_cmp = {}

                # Bisección
                try:
                    res_bis = biseccion(f_fn, a_cmp, b_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    resultados_cmp["Bisección"] = res_bis
                except Exception as e:
                    st.warning(f"Bisección falló: {e}")

                # Newton-Raphson
                try:
                    if expr_df.strip():
                        df_fn = make_fn(expr_df)
                    else:
                        import sympy
                        x_sym = sympy.Symbol('x')
                        MATH_SYMPY_NS = {
                            "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
                            "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                            "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
                            "sqrt": sympy.sqrt, "abs": sympy.Abs,
                        }
                        parsed = sympy.sympify(norm(expr_f).replace("ln(", "log("), locals=MATH_SYMPY_NS)
                        derivada_sym = sympy.diff(parsed, x_sym)
                        df_fn = sympy.lambdify(x_sym, derivada_sym, "math")
                    res_nr = newton_raphson(f_fn, df_fn, x0_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    resultados_cmp["Newton-Raphson"] = res_nr
                except Exception as e:
                    st.warning(f"Newton-Raphson falló: {e}")

                # Secante
                try:
                    res_sec = secante(f_fn, x0_cmp, x1_cmp, tol=tol_cmp, max_iter=int(mi_cmp))
                    resultados_cmp["Secante"] = res_sec
                except Exception as e:
                    st.warning(f"Secante falló: {e}")

                if resultados_cmp:
                    # Tabla comparativa
                    with st.container(border=True):
                        st.markdown("### Tabla Comparativa")
                        tabla_rows = []
                        for nombre, r in resultados_cmp.items():
                            err_final = abs(f_fn(r["raiz"])) if r["raiz"] is not None else None
                            tabla_rows.append({
                                "Método": nombre,
                                "Raíz": f"{r['raiz']:.12f}",
                                "Iteraciones": r["iteraciones"],
                                "|f(raíz)|": f"{err_final:.4e}" if err_final is not None else "N/A",
                                "Convergió": "✅" if r["convergio"] else "❌",
                            })
                        st.dataframe(pd.DataFrame(tabla_rows), hide_index=True, use_container_width=True)

                    # Gráfico de comparación de convergencia
                    with st.container(border=True):
                        st.markdown("### Convergencia Comparada (log₁₀ error)")
                        all_series = []
                        for nombre, r in resultados_cmp.items():
                            for row in r["historial"]:
                                if "error" in row and row["error"] and row["error"] > 0:
                                    all_series.append({
                                        "iter": row["iter"],
                                        "log₁₀(error)": math.log10(row["error"]),
                                        "Método": nombre
                                    })
                        if all_series:
                            df_cmp = pd.DataFrame(all_series)
                            chart_cmp = alt.Chart(df_cmp).mark_line(point=True).encode(
                                x=alt.X("iter:Q", title="Iteración"),
                                y=alt.Y("log₁₀(error):Q", title="log₁₀(|error|)"),
                                color=alt.Color("Método:N"),
                                tooltip=["iter", "Método", "log₁₀(error)"]
                            ).properties(height=350, title="Velocidad de Convergencia por Método")
                            st.altair_chart(chart_cmp, use_container_width=True)
                            st.caption("La pendiente más empinada indica convergencia más rápida.")

            except Exception as e:
                st.error(str(e))
    with side_col:
        st.markdown("### Comparador")
        st.caption("Ejecuta los 3 métodos principales sobre la misma función y compara lado a lado la velocidad de convergencia y precisión final.")
        st.latex(r"p \approx \frac{\ln|e_{n+1}/e_n|}{\ln|e_n/e_{n-1}|}")
        st.caption("p≈1: Bisección | p≈1.62: Secante | p≈2: Newton")


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
            [
                "Función continua (ingreso f(x) y h)",
                "Tabla de datos discretos",
                "Interpolación Newton — diferencias divididas (cualquier espaciado en x)",
            ],
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
                                f"Error abs = {res['error_d1']:.8e}"
                            )
                        if exacta_d2 is not None:
                            st.caption(
                                f"Valor exacto f''(x₀) = {exacta_d2:.10f}  →  "
                                f"Error abs = {res['error_d2']:.8e}"
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
        elif modo == "Tabla de datos discretos":
            with st.container(border=True):
                st.subheader("Diferencias Finitas — Tabla de datos discretos")
                st.caption(
                    "Pares x,y (coma decimal o expresiones: pi, pi/2, sqrt(2), etc.). "
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
                        xs_t.append(parse_number_cell(partes[0]))
                        ys_t.append(parse_number_cell(partes[1]))

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

        # ── MODO INTERPOLACIÓN NEWTON ───────────────────────────────────
        elif modo == "Interpolación Newton — diferencias divididas (cualquier espaciado en x)":
            with st.container(border=True):
                st.subheader("Interpolación de Newton (diferencias divididas)")
                st.caption(
                    "x distintos en orden creciente; el paso puede variar (p. ej. 0, π/4, π/2, π). "
                    "Se obtiene el triángulo de diferencias divididas y P(x) en factores y en potencias."
                )
                with st.expander("Fórmulas", expanded=False):
                    st.latex(
                        r"f[x_i,\ldots,x_{i+k}]="
                        r"\frac{f[x_{i+1},\ldots,x_{i+k}]-f[x_i,\ldots,x_{i+k-1}]}{x_{i+k}-x_i}"
                    )
                    st.latex(
                        r"P(x)=f[x_0]+f[x_0,x_1](x-x_0)+f[x_0,x_1,x_2](x-x_0)(x-x_1)+\cdots"
                    )
                    st.caption("Si los x son equiespaciados, esto coincide con la forma de diferencias progresivas.")
                with st.form("dif_newton_interp"):
                    puntos_n = st.text_area(
                        "Puntos (x, y) — una fila por nodo; expresiones: pi, pi/4, sqrt(2)/2, …",
                        value="0, 0\npi/4, sqrt(2)/2\npi/2, 1\npi, 0",
                        height=140,
                        key="newton_xy",
                    )
                    x_eval_str = st.text_input("Evaluar P(x) en x = (opcional)", "", help="Deja en blanco si sólo quieres calcular el polinomio.")
                    run_n = st.form_submit_button("Calcular polinomio", type="primary")

            if run_n:
                try:
                    lineas = [l.strip() for l in puntos_n.replace(";", "\n").splitlines() if l.strip()]
                    xs_n, ys_n = [], []
                    incognitas = []
                    for linea in lineas:
                        partes = linea.split(",")
                        if len(partes) != 2:
                            raise ValueError(f"Formato incorrecto en línea: '{linea}'. Usa 'x, y'.")
                        
                        xv = parse_number_cell(partes[0])
                        y_str = partes[1].strip()
                        try:
                            yv = parse_number_cell(y_str)
                            xs_n.append(xv)
                            ys_n.append(yv)
                        except ValueError:
                            # Considerarlo una incógnita (e.g. 'k')
                            incognitas.append((xv, y_str))

                    if len(xs_n) < 2:
                        raise ValueError(f"Se necesitan al menos dos nodos numéricos para interpolar (tienes {len(xs_n)}).")

                    res_n = interpolacion_newton_divididas(xs_n, ys_n)

                    with st.container(border=True):
                        st.markdown("### Tabla de diferencias divididas")
                        c1, c2 = st.columns(2)
                        c1.metric("x₀ (primer nodo)", f"{res_n['x0']:.10g}")
                        c2.metric(
                            "Nodos equiespaciados",
                            "sí" if res_n["nodos_equiespaciados"] else "no",
                        )
                        df_tri = pd.DataFrame(res_n["filas_tabla"])
                        st.dataframe(df_tri, hide_index=True, use_container_width=True)
                        st.caption("Columnas div_k = orden k de diferencia dividida en la fila correspondiente.")

                    with st.container(border=True):
                        st.markdown("### Polinomio (forma Newton en factores)")
                        st.latex(r"P(x) = " + res_n["latex_newton_divididas"])
                        st.markdown("### Polinomio en x (expandido, potencias)")
                        st.latex(r"P(x) = " + res_n["latex_polinomio"])
                        st.markdown("### Polinomio en Python (para copiar a Función)")
                        st.text_area(
                            "Expresión en Python",
                            value=res_n.get("python_polinomio", ""),
                            height=68,
                            key="python_copy_newton",
                        )

                    coefs = res_n["coefs_potencias"]

                    def _eval_poly(xv: float) -> float:
                        return sum(coefs[i] * (xv ** i) for i in range(len(coefs)))

                    # Si hubo incógnitas (e.g. 'k' en un nodo x), las calculamos automáticamente
                    pts_incognitas_grafica = []
                    if incognitas:
                        for idx_inc, (xv, nombre) in enumerate(incognitas):
                            val_pred = _eval_poly(xv)
                            st.success(f"**Incógnita resuelta:** Al evaluar $P({xv})$, obtenemos que ${nombre} \\approx {val_pred:.15g}$")
                            pts_incognitas_grafica.append({"x": xv, "y": val_pred, "tipo": f"Incógnita ({nombre})"})

                    # Evaluar P(x) manual
                    if x_eval_str.strip():
                        try:
                            val_x_eval = parse_number_cell(x_eval_str)
                            val_p = _eval_poly(val_x_eval)
                            st.info(f"**Evaluación manual:** $P({val_x_eval}) \\approx {val_p:.15g}$")
                        except Exception as e:
                            st.warning(f"No se pudo evaluar x = {x_eval_str}: {e}")

                    with st.container(border=True):
                        st.markdown("### Verificación en los nodos")
                        ver_rows = []
                        for xv, yv in zip(xs_n, ys_n):
                            pv = _eval_poly(xv)
                            ver_rows.append({
                                "x": xv,
                                "y dado": yv,
                                "P(x)": pv,
                                "error": abs(pv - yv),
                            })
                        st.dataframe(pd.DataFrame(ver_rows), hide_index=True, use_container_width=True)

                    xa_n = min(xs_n)
                    xb_n = max(xs_n)
                    marg = max((xb_n - xa_n) * 0.25, 0.5)
                    xs_plot = []
                    ys_plot = []
                    npt = 300
                    for k in range(npt):
                        xv = xa_n - marg + (k / max(npt - 1, 1)) * (xb_n - xa_n + 2 * marg)
                        xs_plot.append(xv)
                        ys_plot.append(_eval_poly(xv))
                    df_curve = pd.DataFrame({"x": xs_plot, "y": ys_plot})
                    
                    df_pts_conocidos = pd.DataFrame([{"x": x, "y": y, "tipo": "Dato conocido"} for x, y in zip(xs_n, ys_n)])
                    df_pts_totales = pd.concat([df_pts_conocidos, pd.DataFrame(pts_incognitas_grafica)]) if pts_incognitas_grafica else df_pts_conocidos

                    curva_p = (
                        alt.Chart(df_curve)
                        .mark_line(color="#7c3aed", strokeWidth=2)
                        .encode(x="x:Q", y="y:Q")
                    )
                    
                    puntos_p = (
                        alt.Chart(df_pts_totales)
                        .mark_point(size=150, filled=True)
                        .encode(
                            x="x:Q", 
                            y="y:Q", 
                            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Dato conocido", "Incógnita (k)"], range=["#f97316", "#10b981"])),
                            tooltip=["x", "y", "tipo"]
                        )
                    )
                    with st.container(border=True):
                        st.markdown("### P(x) y nodos")
                        st.altair_chart(
                            (curva_p + puntos_p).properties(height=320).interactive(),
                            use_container_width=True,
                        )

                except Exception as e:
                    st.error(str(e))

    with side_col:
        _formulas_panel("Diferencias Finitas")


# ── INTEGRACIÓN NUMÉRICA ───────────────────────────────────────────────────
if algoritmo == "Integración Numérica":
    import numpy as np # Local import for np array
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
                    ndec = st.number_input(
                        "Decimales a mostrar",
                        min_value=6,
                        max_value=24,
                        value=16,
                        step=1,
                        key="int_ndec",
                    )
                
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
                            I_ref = None
                            err_quad = None
                            try:
                                I_ref, err_quad = integral_referencia(f_fn, a, b)
                            except Exception as qe:
                                st.warning(
                                    "No se pudo obtener la integral de referencia (quad): "
                                    f"{qe}"
                                )

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
                                    rows.append(
                                        rf"I_{{\mathrm{{ref}}}} &\approx {fmt_f.format(I_ref)}"
                                    )
                                    e_trunc = I_ref - res["valor"]
                                    rows.append(
                                        rf"E_{{\mathrm{{trunc}}}} = I_{{\mathrm{{ref}}}} - I_{{\mathrm{{aprox}}}} "
                                        rf"&\approx {fmt_f.format(e_trunc)}"
                                    )
                                    rows.append(
                                        rf"\left| E_{{\mathrm{{trunc}}}} \right| "
                                        rf"&\approx {fmt_f.format(abs(e_trunc))}"
                                    )
                                if err_quad is not None and math.isfinite(err_quad):
                                    rows.append(
                                        rf"\varepsilon_{{\mathrm{{quad}}}} "
                                        rf"&\approx {fmt_f.format(err_quad)}"
                                    )
                                st.latex(
                                    "\\begin{aligned} "
                                    + " \\\\ ".join(rows)
                                    + " \\end{aligned}"
                                )
                                st.caption(
                                    "LaTeX del integrando vía SymPy; la integral numérica es la del método elegido "
                                    "(referencia con quad o trapecio denso si no hay SciPy)."
                                )

                            with st.container(border=True):
                                st.markdown(f"### {res['metodo']}")
                                cols = st.columns(4)
                                cols[0].metric(
                                    "∫ f (referencia)",
                                    fmt_f.format(I_ref) if I_ref is not None else "—",
                                    help="Cuadratura adaptativa (scipy.integrate.quad).",
                                )
                                cols[1].metric(
                                    "Integral aproximada",
                                    fmt_f.format(res["valor"]),
                                    help="Newton-Cotes compuesto con el método elegido.",
                                )
                                cols[2].metric("h (paso)", f"{res['h']:.{min(nd, 10)}f}")
                                cols[3].metric("Nodos", len(res["x_vals"]))

                            if I_ref is not None:
                                e_trunc = I_ref - res["valor"]
                                c_err = st.columns(3)
                                c_err[0].metric(
                                    "Error truncamiento |E|",
                                    fmt_f.format(abs(e_trunc)),
                                    help="|I_ref − I_aprox|; I_ref actúa como valor ‘verdadero’.",
                                )
                                c_err[1].metric(
                                    "E_trunc (con signo)",
                                    f"{e_trunc:+.{nd}f}",
                                    help="I_ref − I_aprox (positivo si la aproximación subestima la integral).",
                                )
                                c_err[2].metric(
                                    "ε_quad (referencia)",
                                    fmt_f.format(err_quad)
                                    if err_quad is not None and math.isfinite(err_quad)
                                    else "—",
                                    help="Cota estimada del error numérico de quad (no es error de truncamiento del método mostrado).",
                                )

                            with st.container(border=True):
                                st.markdown("### Área de Aproximación")
                                
                                xs = np.linspace(a, b, 600)
                                ys = []
                                for xv in xs:
                                    try: ys.append(float(f_fn(xv)))
                                    except: ys.append(float("nan"))
                                
                                df_smooth = pd.DataFrame({"x": xs, "f(x)": ys})
                                curva = alt.Chart(df_smooth).mark_line(color="#2563eb", strokeWidth=3).encode(
                                    x=alt.X("x:Q", title="x"), 
                                    y=alt.Y("f(x):Q", title="f(x)"), 
                                    tooltip=["x", "f(x)"]
                                )
                                
                                df_approx = pd.DataFrame({"x": res["x_vals"], "f(x)": res["y_vals"]})
                                if metodo == "Rectángulo (Medio)":
                                    aprox = alt.Chart(df_approx).mark_bar(opacity=0.4, color="#ef4444", size=(1000/int(n))*0.5).encode(x="x:Q", y="f(x):Q")
                                    px = alt.Chart(df_approx).mark_point(color="#dc2626", filled=True, size=60).encode(x="x:Q", y="f(x):Q")
                                else:
                                    aprox = alt.Chart(df_approx).mark_area(opacity=0.4, color="#ef4444").encode(x="x:Q", y="f(x):Q")
                                    px = alt.Chart(df_approx).mark_point(color="#dc2626", filled=True, size=60).encode(x="x:Q", y="f(x):Q")
                                
                                grafico = (curva + aprox + px).properties(height=350).interactive()
                                st.altair_chart(grafico, use_container_width=True)

                except Exception as e:
                    st.error(str(e))
    
    with side_col:
        _formulas_panel("Integración Numérica")


# ── MONTE CARLO ────────────────────────────────────────────────────────────
if algoritmo == "Monte Carlo":
    main_col, side_col = st.columns([2.3, 1.0], gap="large")
    with main_col:
        modo = st.radio("Modo de Simulación", ["Estimación de π", "Integración 1D", "Integración 2D", "Convergencia Progresiva", "Histograma CLT", "Comparar 2 Integrales", "Monty Hall (Juego)"], horizontal=True)

        if modo == "Estimación de π":
            with st.container(border=True):
                st.subheader("Monte Carlo: Aciertos en un Círculo")
                st.caption("A medida que N crece, la proporción de aciertos dentro del radio unitario aproxima a π/4.")
                with st.form("mc_pi"):
                    c1, c2 = st.columns(2)
                    with c1:
                        n_puntos = st.number_input("Número de Puntos (N)", value=100000, step=10000)
                        animar_pi = st.checkbox("Animar caída de puntos", value=False)
                    with c2:
                        semilla = st.number_input("Semilla Aleatoria (opcional)", value=42, min_value=0, step=1)
                        usar_semilla = st.checkbox("Fijar semilla para reproducibilidad", value=True)
                    run_pi = st.form_submit_button("Simular", type="primary")

            if run_pi:
                seed = int(semilla) if usar_semilla else None
                res_pi = estimar_pi(int(n_puntos), seed)
                with st.container(border=True):
                    cols = st.columns(3)
                    cols[0].metric("Puntos Disparados (N)", f"{res_pi['num_puntos']}")
                    cols[1].metric("Puntos dentro (Acertados)", f"{res_pi['puntos_dentro']}")
                    cols[2].metric("Aproximación de π", f"{res_pi['pi_estimado']:.8f}")

                with st.container(border=True):
                    st.markdown("#### Resultado Detallado")
                    import math as _m
                    st.latex(rf"\pi \approx 4 \times \frac{{{res_pi['puntos_dentro']}}}{{{res_pi['num_puntos']}}} = {res_pi['pi_estimado']:.10f}")
                    st.latex(rf"\left|\,\pi_{{\text{{estimado}}}} - \pi_{{\text{{real}}}}\,\right| = {res_pi['error_vs_pi']:.10f}")
                    st.latex(rf"\text{{Error Estándar}} = {res_pi['error_estandar']:.10f}")
                    st.latex(rf"\text{{IC}}^{{95\%}} = \left[{res_pi['pi_estimado'] - 1.96*res_pi['error_estandar']:.10f},\ {res_pi['pi_estimado'] + 1.96*res_pi['error_estandar']:.10f}\right]")

                pts = res_pi["puntos_grafica"]
                if pts:
                    import pandas as pd
                    import time
                    st.caption(f"Mostrando los primeros {len(pts)} puntos generados:")
                    df = pd.DataFrame(pts)
                    
                    if animar_pi:
                        placeholder = st.empty()
                        num_frames = 20
                        chunk_size = max(1, len(pts) // num_frames)
                        for i in range(1, num_frames + 1):
                            idx = i * chunk_size if i < num_frames else len(pts)
                            df_chunk = df.iloc[:idx]
                            chart = alt.Chart(df_chunk).mark_circle(size=15).encode(
                                x=alt.X('x:Q', scale=alt.Scale(domain=[-1, 1])),
                                y=alt.Y('y:Q', scale=alt.Scale(domain=[-1, 1])),
                                color=alt.Color('estado:N', scale=alt.Scale(domain=["Dentro", "Fuera"], range=["#22c55e", "#ef4444"]))
                            ).properties(height=400, width=400)
                            placeholder.altair_chart(chart, use_container_width=True)
                            time.sleep(0.1)
                        # Volver interactivo al terminar
                        chart_final = alt.Chart(df).mark_circle(size=15).encode(
                            x=alt.X('x:Q', scale=alt.Scale(domain=[-1, 1])),
                            y=alt.Y('y:Q', scale=alt.Scale(domain=[-1, 1])),
                            color=alt.Color('estado:N', scale=alt.Scale(domain=["Dentro", "Fuera"], range=["#22c55e", "#ef4444"]))
                        ).properties(height=400, width=400).interactive()
                        placeholder.altair_chart(chart_final, use_container_width=True)
                    else:
                        chart = alt.Chart(df).mark_circle(size=15).encode(
                            x=alt.X('x:Q', scale=alt.Scale(domain=[-1, 1])),
                            y=alt.Y('y:Q', scale=alt.Scale(domain=[-1, 1])),
                            color=alt.Color('estado:N', scale=alt.Scale(domain=["Dentro", "Fuera"], range=["#22c55e", "#ef4444"]))
                        ).properties(height=400, width=400).interactive()
                        st.altair_chart(chart, use_container_width=True)

        elif modo == "Integración 1D":
            with st.container(border=True):
                st.subheader("Monte Carlo: Integración 1D")
                st.caption("Evaluación estocástica. Devuelve un Intervalo de Confianza probabilístico, no una cota determinista.")
                with st.form("mc_1d"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f = st.text_input("f(x)", "exp(-x**2)", key="mc_f1")
                        a_str = st.text_input("Límite inferior a", "0")
                        b_str = st.text_input("Límite superior b", "2")
                    with c2:
                        n_puntos = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000)
                        confianza = st.selectbox("Intervalo de Confianza", ["90%", "95%", "99%"], index=1)
                        animar_1d = st.checkbox("Animar caída de puntos", value=False)
                        semilla = st.number_input("Semilla Aleatoria", value=42, min_value=0, step=1)
                        usar_semilla = st.checkbox("Fijar semilla", value=True)
                    run_1d = st.form_submit_button("Integrar", type="primary")

            if run_1d:
                try:
                    f_fn = make_fn(expr_f)
                    a = parse_number_cell(a_str)
                    b = parse_number_cell(b_str)
                    seed = int(semilla) if usar_semilla else None
                    
                    res_1d = integracion_1d_mc(f_fn, a, b, int(n_puntos), confianza, seed)
                    
                    with st.container(border=True):
                        st.success(f"Cálculo estadístico finalizado tras {n_puntos} muestras evaluadas.")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Integral Estimada Î", f"{res_1d['integral']:.8f}")
                        m2.metric("Margen de Error (±)", f"{res_1d['margen_error']:.8f}")
                        m3.metric(f"IC al {confianza}", "Ver Tabla Abajo")
                        
                        st.markdown("#### Resultado Detallado")
                        st.latex(rf"\hat{{I}} = (b-a) \cdot \bar{{f}} = {res_1d['integral']:.10f}")
                        st.latex(rf"\sigma = {res_1d['sigma']:.10f}")
                        st.latex(rf"\text{{Error Estándar}} = \frac{{\sigma}}{{\sqrt{{N}}}} = {res_1d['error_estandar']:.10f}")
                        st.latex(rf"\text{{Margen de Error}} = Z \cdot EE \cdot (b-a) = {res_1d['margen_error']:.10f}")
                        conf_latex = confianza.replace("%", r"\%")
                        st.latex(rf"\text{{IC}}^{{{conf_latex}}} = \left[ {res_1d['ic_inferior']:.10f},\ {res_1d['ic_superior']:.10f} \right]")
                        
                    graf = res_1d.get("puntos_grafica")
                    if graf:
                        import time
                        st.caption(f"Mostrando frecuencias y distribución de muestras evaluadas en f(x_i) (max {len(graf)}):")
                        df = pd.DataFrame(graf)
                        
                        if animar_1d:
                            placeholder = st.empty()
                            num_frames = 20
                            chunk_size = max(1, len(graf) // num_frames)
                            for i in range(1, num_frames + 1):
                                idx = i * chunk_size if i < num_frames else len(graf)
                                df_chunk = df.iloc[:idx]
                                chart = alt.Chart(df_chunk).mark_circle(opacity=0.3, size=20, color="#3b82f6").encode(
                                    x=alt.X('x:Q', scale=alt.Scale(domain=[a,b])),
                                    y=alt.Y('y:Q', title="f(x)")
                                ).properties(height=350)
                                placeholder.altair_chart(chart, use_container_width=True)
                                time.sleep(0.1)
                            chart_final = alt.Chart(df).mark_circle(opacity=0.3, size=20, color="#3b82f6").encode(
                                x=alt.X('x:Q', scale=alt.Scale(domain=[a,b])),
                                y=alt.Y('y:Q', title="f(x)")
                            ).properties(height=350).interactive()
                            placeholder.altair_chart(chart_final, use_container_width=True)
                        else:
                            chart = alt.Chart(df).mark_circle(opacity=0.3, size=20, color="#3b82f6").encode(
                                x=alt.X('x:Q', scale=alt.Scale(domain=[a,b])),
                                y=alt.Y('y:Q', title="f(x)")
                            ).properties(height=350).interactive()
                            st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"Error procesando: {e}")

        elif modo == "Integración 2D":
            with st.container(border=True):
                st.subheader("Monte Carlo: Integración 2D Multivariable")
                st.caption("Excelente para evadir la 'maldición de la dimensionalidad' en volúmenes y áreas 2D.")
                with st.form("mc_2d"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f2 = st.text_input("f(x, y)", "sin(x) * cos(y)", help="Usa variables 'x' e 'y'.")
                        ax_str = st.text_input("Límite x inferior (a)", "0")
                        bx_str = st.text_input("Límite x superior (b)", "pi/2")
                    with c2:
                        cy_str = st.text_input("Límite y inferior (c)", "0")
                        dy_str = st.text_input("Límite y superior (d)", "pi/2")
                        n_puntos = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000)
                        confianza = st.selectbox("Intervalo de Confianza", ["90%", "95%", "99%"], index=2)
                        semilla = st.number_input("Semilla Aleatoria", value=42, min_value=0, step=1)
                        usar_semilla = st.checkbox("Fijar semilla", value=True)
                    run_2d = st.form_submit_button("Calcular Volumen/Integral", type="primary")

            if run_2d:
                try:
                    f_2d_fn = make_fn_2d(expr_f2)
                    a = parse_number_cell(ax_str)
                    b = parse_number_cell(bx_str)
                    c = parse_number_cell(cy_str)
                    d = parse_number_cell(dy_str)
                    seed = int(semilla) if usar_semilla else None
                    
                    res_2d = integracion_2d_mc(f_2d_fn, a, b, c, d, int(n_puntos), confianza, seed)
                    
                    with st.container(border=True):
                        st.success(f"Cálculo multidimensional finalizado tras {n_puntos} estimaciones sobre la grilla Z.")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Integral Estimada Î", f"{res_2d['integral']:.8f}")
                        m2.metric("Margen de Error (±)", f"{res_2d['margen_error']:.8f}")
                        m3.metric(f"IC al {confianza}", "Ver Tabla Abajo")

                        st.markdown("#### Resultado Detallado")
                        st.latex(rf"\hat{{I}} = \text{{Área}} \cdot \bar{{f}} = {res_2d['integral']:.10f}")
                        st.latex(rf"\sigma = {res_2d['sigma']:.10f}")
                        st.latex(rf"\text{{Error Estándar}} = \frac{{\sigma}}{{\sqrt{{N}}}} = {res_2d['error_estandar']:.10f}")
                        st.latex(rf"\text{{Margen de Error}} = Z \cdot EE \cdot \text{{Área}} = {res_2d['margen_error']:.10f}")
                        conf_latex = confianza.replace("%", r"\%")
                        st.latex(rf"\text{{IC}}^{{{conf_latex}}} = \left[ {res_2d['ic_inferior']:.10f},\ {res_2d['ic_superior']:.10f} \right]")
                except Exception as e:
                    st.error(f"Error procesando la integral iterativa: {e}")

        elif modo == "Convergencia Progresiva":
            with st.container(border=True):
                st.subheader("Convergencia Progresiva de Monte Carlo")
                st.caption("Observá cómo la estimación se estabiliza conforme N crece. La banda del IC se achica.")
                with st.form("mc_conv"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f_conv = st.text_input("f(x)", "x**2", key="mc_conv_f")
                        a_conv = st.text_input("Límite inferior a", "0", key="mc_conv_a")
                        b_conv = st.text_input("Límite superior b", "2", key="mc_conv_b")
                    with c2:
                        n_conv = st.number_input("Número de Evaluaciones (N)", value=100000, step=10000, key="mc_conv_n")
                        conf_conv = st.selectbox("IC", ["90%", "95%", "99%"], index=1, key="mc_conv_ic")
                        semilla_conv = st.number_input("Semilla", value=42, min_value=0, step=1, key="mc_conv_s")
                        usar_semilla_conv = st.checkbox("Fijar semilla", value=True, key="mc_conv_fix")
                    run_conv = st.form_submit_button("Graficar Convergencia", type="primary")

            if run_conv:
                try:
                    f_fn = make_fn(expr_f_conv)
                    a = parse_number_cell(a_conv)
                    b = parse_number_cell(b_conv)
                    seed = int(semilla_conv) if usar_semilla_conv else None
                    res_conv = integracion_1d_mc(f_fn, a, b, int(n_conv), conf_conv, seed)
                    snaps = res_conv.get("snapshots", [])
                    if snaps:
                        with st.container(border=True):
                            st.markdown("### Evolución de la Estimación")
                            df_snap = pd.DataFrame(snaps)
                            linea = alt.Chart(df_snap).mark_line(color="#2563eb", strokeWidth=2).encode(
                                x=alt.X("n:Q", title="Número de muestras (N)"),
                                y=alt.Y("integral:Q", title="Integral Estimada")
                            )
                            banda = alt.Chart(df_snap).mark_area(opacity=0.2, color="#3b82f6").encode(
                                x="n:Q",
                                y="ic_inf:Q",
                                y2="ic_sup:Q"
                            )
                            valor_final = alt.Chart(pd.DataFrame([{"y": res_conv["integral"]}])).mark_rule(
                                color="#dc2626", strokeDash=[5, 3], strokeWidth=2
                            ).encode(y="y:Q")
                            chart = (banda + linea + valor_final).properties(
                                height=400, title="Convergencia Progresiva con Banda de IC"
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                            st.caption("Línea azul: estimación parcial. Banda: IC progresivo. Línea roja: valor final.")
                            st.latex(rf"\hat{{I}}_{{\text{{final}}}} = {res_conv['integral']:.10f}")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif modo == "Histograma CLT":
            with st.container(border=True):
                st.subheader("Histograma: Teorema Central del Límite")
                st.caption("Ejecuta K simulaciones independientes y grafica la distribución de resultados. Demuestra visualmente el TCL.")
                with st.form("mc_clt"):
                    c1, c2 = st.columns(2)
                    with c1:
                        expr_f_clt = st.text_input("f(x)", "x**2", key="mc_clt_f")
                        a_clt = st.text_input("a", "0", key="mc_clt_a")
                        b_clt = st.text_input("b", "2", key="mc_clt_b")
                    with c2:
                        n_clt = st.number_input("N por corrida", value=5000, step=1000, key="mc_clt_n")
                        k_clt = st.number_input("Número de corridas (K)", value=200, step=50, key="mc_clt_k")
                    run_clt = st.form_submit_button("Simular K corridas", type="primary")

            if run_clt:
                try:
                    f_fn = make_fn(expr_f_clt)
                    a = parse_number_cell(a_clt)
                    b = parse_number_cell(b_clt)
                    with st.spinner(f"Ejecutando {k_clt} simulaciones MC..."):
                        resultados = multi_run_1d(f_fn, a, b, int(n_clt), int(k_clt))
                    with st.container(border=True):
                        st.markdown("### Distribución de Resultados")
                        import numpy as np
                        media = np.mean(resultados)
                        desv = np.std(resultados, ddof=1)
                        st.latex(rf"\bar{{I}} = {media:.10f}")
                        st.latex(rf"s = {desv:.10f}")
                        st.latex(rf"\text{{K corridas}} = {int(k_clt)}")

                        df_hist = pd.DataFrame({"Integral Estimada": resultados})
                        hist_chart = alt.Chart(df_hist).mark_bar(opacity=0.7, color="#6366f1").encode(
                            alt.X("Integral Estimada:Q", bin=alt.Bin(maxbins=40), title="Valor de la Integral"),
                            y=alt.Y("count()", title="Frecuencia")
                        ).properties(height=350, title=f"Histograma de {int(k_clt)} corridas MC (N={int(n_clt)} c/u)")

                        media_rule = alt.Chart(pd.DataFrame([{"x": media}])).mark_rule(
                            color="#dc2626", strokeWidth=2, strokeDash=[5,3]
                        ).encode(x="x:Q")

                        st.altair_chart(hist_chart + media_rule, use_container_width=True)
                        st.caption("La distribución se aproxima a una Normal (Teorema Central del Límite). Línea roja: media.")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif modo == "Comparar 2 Integrales":
            with st.container(border=True):
                st.subheader("Comparar dos Integrales con Monte Carlo")
                st.caption("Evalúa dos integrales distintas con los mismos parámetros y compara sus resultados e intervalos de confianza lado a lado.")
                with st.form("mc_cmp2"):
                    st.markdown("##### Integral A")
                    ca1, ca2, ca3 = st.columns(3)
                    with ca1:
                        expr_a = st.text_input("f(x) — Integral A", "x**2", key="mc_a_f")
                    with ca2:
                        a_a = st.text_input("a (A)", "0", key="mc_a_a")
                        b_a = st.text_input("b (A)", "2", key="mc_a_b")
                    with ca3:
                        label_a = st.text_input("Etiqueta A", "Integral A", key="mc_a_label")

                    st.markdown("##### Integral B")
                    cb1, cb2, cb3 = st.columns(3)
                    with cb1:
                        expr_b = st.text_input("f(x) — Integral B", "sin(x)", key="mc_b_f")
                    with cb2:
                        a_b = st.text_input("a (B)", "0", key="mc_b_a")
                        b_b = st.text_input("b (B)", "pi", key="mc_b_b")
                    with cb3:
                        label_b = st.text_input("Etiqueta B", "Integral B", key="mc_b_label")

                    st.markdown("##### Parámetros comunes")
                    cp1, cp2, cp3 = st.columns(3)
                    with cp1:
                        n_cmp2 = st.number_input("N evaluaciones", value=100000, step=10000, key="mc_cmp2_n")
                    with cp2:
                        conf_cmp2 = st.selectbox("IC", ["90%", "95%", "99%"], index=1, key="mc_cmp2_ic")
                    with cp3:
                        semilla_cmp2 = st.number_input("Semilla", value=42, min_value=0, step=1, key="mc_cmp2_s")
                        usar_semilla_cmp2 = st.checkbox("Fijar semilla", value=True, key="mc_cmp2_fix")
                    run_cmp2 = st.form_submit_button("Comparar Integrales", type="primary")

            if run_cmp2:
                try:
                    seed = int(semilla_cmp2) if usar_semilla_cmp2 else None
                    fa = make_fn(expr_a)
                    fb = make_fn(expr_b)
                    aa = parse_number_cell(a_a); ba = parse_number_cell(b_a)
                    ab = parse_number_cell(a_b); bb = parse_number_cell(b_b)

                    res_a = integracion_1d_mc(fa, aa, ba, int(n_cmp2), conf_cmp2, seed)
                    res_b = integracion_1d_mc(fb, ab, bb, int(n_cmp2), conf_cmp2, seed)

                    with st.container(border=True):
                        st.markdown("### Tabla Comparativa")
                        st.table(pd.DataFrame([
                            {
                                "Integral": label_a,
                                "f(x)": expr_a,
                                "[a, b]": f"[{aa}, {ba}]",
                                "\u00ce": f"{res_a['integral']:.10f}",
                                "Margen \u00b1": f"{res_a['margen_error']:.10f}",
                                f"IC inf ({conf_cmp2})": f"{res_a['ic_inferior']:.10f}",
                                f"IC sup ({conf_cmp2})": f"{res_a['ic_superior']:.10f}",
                            },
                            {
                                "Integral": label_b,
                                "f(x)": expr_b,
                                "[a, b]": f"[{ab}, {bb}]",
                                "\u00ce": f"{res_b['integral']:.10f}",
                                "Margen \u00b1": f"{res_b['margen_error']:.10f}",
                                f"IC inf ({conf_cmp2})": f"{res_b['ic_inferior']:.10f}",
                                f"IC sup ({conf_cmp2})": f"{res_b['ic_superior']:.10f}",
                            },
                        ]))

                    with st.container(border=True):
                        st.markdown("### Resultados en LaTeX")
                        col_la, col_lb = st.columns(2)
                        with col_la:
                            st.markdown(f"**{label_a}**")
                            st.latex(rf"\hat{{I}}_A = {res_a['integral']:.10f}")
                            st.latex(rf"\sigma_A = {res_a['sigma']:.10f}")
                            conf_l = conf_cmp2.replace('%', r'\%')
                            st.latex(rf"IC_A^{{{conf_l}}} = \left[{res_a['ic_inferior']:.10f},\ {res_a['ic_superior']:.10f}\right]")
                        with col_lb:
                            st.markdown(f"**{label_b}**")
                            st.latex(rf"\hat{{I}}_B = {res_b['integral']:.10f}")
                            st.latex(rf"\sigma_B = {res_b['sigma']:.10f}")
                            st.latex(rf"IC_B^{{{conf_l}}} = \left[{res_b['ic_inferior']:.10f},\ {res_b['ic_superior']:.10f}\right]")

                    # Grafico de convergencia comparado
                    snaps_a = res_a.get("snapshots", [])
                    snaps_b = res_b.get("snapshots", [])
                    if snaps_a and snaps_b:
                        with st.container(border=True):
                            st.markdown("### Convergencia Comparada")
                            df_a = pd.DataFrame(snaps_a); df_a["serie"] = label_a
                            df_b = pd.DataFrame(snaps_b); df_b["serie"] = label_b
                            df_conv = pd.concat([df_a, df_b], ignore_index=True)
                            chart_conv = alt.Chart(df_conv).mark_line(strokeWidth=2).encode(
                                x=alt.X("n:Q", title="N muestras"),
                                y=alt.Y("integral:Q", title="Integral Estimada"),
                                color=alt.Color("serie:N"),
                                tooltip=["n", "serie", "integral"]
                            ).properties(height=380, title="Convergencia progresiva comparada").interactive()
                            st.altair_chart(chart_conv, use_container_width=True)

                    # ── Grafico de las dos funciones + puntos de muestreo ──
                    with st.container(border=True):
                        st.markdown("### Visualización de f(x) y Puntos de Muestreo")
                        st.caption("Curva continua de cada función junto con los puntos aleatoriamente evaluados por Monte Carlo.")

                        import numpy as np

                        # Evaluar curvas continuas
                        def _curva(f, a_v, b_v, etiqueta, color):
                            xs = np.linspace(a_v, b_v, 500)
                            rows = []
                            for xv in xs:
                                try:
                                    y = float(f(xv))
                                    if math.isfinite(y):
                                        rows.append({"x": xv, "y": y, "serie": etiqueta})
                                except Exception:
                                    pass
                            df_c = pd.DataFrame(rows)
                            return alt.Chart(df_c).mark_line(strokeWidth=2.5, color=color).encode(
                                x=alt.X("x:Q", title="x"),
                                y=alt.Y("y:Q", title="f(x)"),
                                tooltip=["x", "y"]
                            )

                        def _puntos(pts_lista, etiqueta, color):
                            df_p = pd.DataFrame(pts_lista)
                            df_p["serie"] = etiqueta
                            return alt.Chart(df_p).mark_circle(size=12, opacity=0.25, color=color).encode(
                                x="x:Q", y="y:Q",
                                tooltip=["x", "y"]
                            )

                        curva_a = _curva(fa, aa, ba, label_a, "#2563eb")
                        curva_b = _curva(fb, ab, bb, label_b, "#dc2626")

                        capas = [curva_a, curva_b]

                        pts_a = res_a.get("puntos_grafica", [])
                        pts_b = res_b.get("puntos_grafica", [])
                        if pts_a:
                            capas.append(_puntos(pts_a, label_a, "#2563eb"))
                        if pts_b:
                            capas.append(_puntos(pts_b, label_b, "#dc2626"))

                        chart_fns = alt.layer(*capas).properties(
                            height=420,
                            title="Funciones integrando y puntos de muestreo MC"
                        ).interactive()
                        st.altair_chart(chart_fns, use_container_width=True)
                        st.caption(f"Azul: {label_a} | Rojo: {label_b} — los puntos son las evaluaciones aleatorias de MC (máx 2000 c/u).")

                except Exception as e:
                    st.error(f"Error: {e}")

        elif modo == "Monty Hall (Juego)":
            with st.container(border=True):
                st.subheader("🐐 El Casino de Monty Hall (Destructor de Intuición)")
                st.caption("Juego matemático clásico: 3 puertas, 1 auto, 2 cabras. ¿Conviene cambiar de puerta después de que el presentador abre una que tiene una cabra? ¡Dejemos que Monte Carlo lo demuestre!")
                
                with st.form("mc_monty_hall"):
                    c1, c2 = st.columns(2)
                    with c1:
                        num_partidas = st.number_input("Número de Partidas a jugar", min_value=100, value=10000, step=1000, key="mh_partidas")
                    with c2:
                        semilla_mh = st.number_input("Semilla Aleatoria", value=42, min_value=0, step=1, key="mh_seed")
                        usar_semilla_mh = st.checkbox("Fijar semilla", value=True, key="mh_usar_seed")
                        animar_mh = st.checkbox("Animar simulación", value=True, key="mh_animar")
                    
                    run_mh = st.form_submit_button("Simular las partidas", type="primary")

            if run_mh:
                try:
                    seed = int(semilla_mh) if usar_semilla_mh else None
                    res_mh = simular_monty_hall(int(num_partidas), semilla=seed)
                    
                    with st.container(border=True):
                        st.markdown("### Resultados Estadísticos")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Partidas Jugadas", f"{res_mh['num_partidas']:,}")
                        
                        win_rate_m = res_mh['tasa_mantener'] * 100
                        col2.metric("Estrategia: MANTENER", f"{win_rate_m:.2f}%", f"{res_mh['wins_mantener']} ganadas", delta_color="off")
                        
                        win_rate_c = res_mh['tasa_cambiar'] * 100
                        delta_color = "normal" if win_rate_c > win_rate_m else "inverse"
                        col3.metric("Estrategia: CAMBIAR", f"{win_rate_c:.2f}%", f"{res_mh['wins_cambiar']} ganadas", delta_color=delta_color)

                    historial = res_mh["historial"]
                    if historial:
                        with st.container(border=True):
                            st.markdown("### La Carrera de Monte Carlo: Cambiar vs Mantener")
                            st.caption("0")
                            
                            import pandas as pd
                            df_h = pd.DataFrame(historial)
                            df_h_melted = df_h.melt("partida", var_name="Estrategia", value_name="Win Rate")
                            
                            df_h_melted["Estrategia"] = df_h_melted["Estrategia"].replace({
                                "win_rate_mantener": "Mantener mi puerta original",
                                "win_rate_cambiar": "Cambiar de puerta"
                            })
                            
                            if animar_mh:
                                import time
                                placeholder = st.empty()
                                # Animar en chunks de 20 cuadros
                                num_frames = min(20, len(historial))
                                chunk_size = max(1, len(historial) // num_frames)
                                
                                for i in range(1, num_frames + 1):
                                    idx = i * chunk_size if i < num_frames else len(historial)
                                    df_chunk = pd.DataFrame(historial[:idx]).melt("partida", var_name="Estrategia", value_name="Win Rate")
                                    df_chunk["Estrategia"] = df_chunk["Estrategia"].replace({
                                        "win_rate_mantener": "Mantener mi puerta original",
                                        "win_rate_cambiar": "Cambiar de puerta"
                                    })
                                    
                                    chart = alt.Chart(df_chunk).mark_line(strokeWidth=3).encode(
                                        x=alt.X("partida:Q", title="Nº de Partida"),
                                        y=alt.Y("Win Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                                        color=alt.Color("Estrategia:N", scale=alt.Scale(domain=["Mantener mi puerta original", "Cambiar de puerta"], range=["#ef4444", "#22c55e"])),
                                    ).properties(height=400)
                                    placeholder.altair_chart(chart, use_container_width=True)
                                    time.sleep(0.1)
                                
                                # Gráfico interactivo final
                                chart_f = alt.Chart(df_h_melted).mark_line(strokeWidth=3).encode(
                                    x=alt.X("partida:Q", title="Nº de Partida"),
                                    y=alt.Y("Win Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                                    color=alt.Color("Estrategia:N", scale=alt.Scale(domain=["Mantener mi puerta original", "Cambiar de puerta"], range=["#ef4444", "#22c55e"])),
                                    tooltip=[alt.Tooltip("partida:Q"), alt.Tooltip("Estrategia:N"), alt.Tooltip("Win Rate:Q", format='.2%')]
                                ).properties(height=400).interactive()
                                placeholder.altair_chart(chart_f, use_container_width=True)
                            else:
                                chart = alt.Chart(df_h_melted).mark_line(strokeWidth=3).encode(
                                    x=alt.X("partida:Q", title="Nº de Partida"),
                                    y=alt.Y("Win Rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
                                    color=alt.Color("Estrategia:N", scale=alt.Scale(domain=["Mantener mi puerta original", "Cambiar de puerta"], range=["#ef4444", "#22c55e"])),
                                    tooltip=[alt.Tooltip("partida:Q"), alt.Tooltip("Estrategia:N"), alt.Tooltip("Win Rate:Q", format='.2%')]
                                ).properties(height=400).interactive()
                                st.altair_chart(chart, use_container_width=True)
                                
                except Exception as e:
                    st.error(f"Error procesando Monty Hall: {e}")

    with side_col:
        _formulas_panel("Monte Carlo")
