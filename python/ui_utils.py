"""
ui_utils.py — Shared utilities for all dashboard UI modules.

Contains:
  - Math namespace & function parsers  (norm, make_fn, make_fn_2d, make_fn_ty)
  - Number/cell parsing                (parse_number_cell)
  - LaTeX formatting helpers           (_latex_float, _latex_bound, _fmt_iter_df,
                                        expr_integrando_latex)
  - Shared Streamlit rendering helpers (_formulas_panel, graficar_fx, mostrar_resultados)
"""
import math

import altair as alt
import pandas as pd
import streamlit as st

# ── Math namespace for eval ────────────────────────────────────────────────────
MATH_NS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "ln": math.log,
    "sqrt": math.sqrt, "abs": abs,
    "cbrt": lambda x: math.copysign(abs(x) ** (1.0 / 3.0), x),
    "pi": math.pi, "e": math.e,
}

MATH_SYMPY_NS_FACTORY = None  # lazy


def _get_sympy_ns():
    import sympy
    return {
        "pi": sympy.pi, "e": sympy.E, "E": sympy.E,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "exp": sympy.exp, "log": sympy.log, "ln": sympy.log,
        "sqrt": sympy.sqrt, "abs": sympy.Abs,
        "cbrt": lambda z: sympy.root(z, 3),
    }


# ── Expression normaliser ──────────────────────────────────────────────────────
def norm(expr: str) -> str:
    return expr.replace("^", "**").strip()


# ── Function parsers ───────────────────────────────────────────────────────────
def make_fn(expr: str):
    """Parse a math expression in x and return a callable f(x) → float."""
    import sympy
    expr = norm(expr).replace("ln(", "log(")
    ns = _get_sympy_ns()
    try:
        parsed_expr = sympy.sympify(expr, locals=ns)
        x_sym = sympy.Symbol("x")
        f_lambdified = sympy.lambdify(x_sym, parsed_expr, "math")

        def f(x):
            try:
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
                return float("nan")

        return f
    except Exception as e:
        raise ValueError(f"Expresión matemática inválida: {e}")


def make_fn_2d(expr: str):
    """Parse a math expression in x, y and return f(x, y) → float."""
    import sympy
    expr = norm(expr).replace("ln(", "log(")
    ns = _get_sympy_ns()
    try:
        parsed_expr = sympy.sympify(expr, locals=ns)
        x_sym, y_sym = sympy.Symbol("x"), sympy.Symbol("y")
        f_lambdified = sympy.lambdify((x_sym, y_sym), parsed_expr, "math")

        def f(xv, yv):
            try:
                val = f_lambdified(xv, yv)
                if isinstance(val, complex) and val.imag == 0:
                    val = val.real
                return float(val)
            except Exception:
                return float("nan")

        return f
    except Exception as e:
        raise ValueError(f"Expresión matemática 2D inválida: {e}")


def make_fn_ty(expr: str):
    """Parse a math expression in t, y and return f(t, y) → float."""
    import sympy
    expr = norm(expr).replace("ln(", "log(")
    ns = _get_sympy_ns()
    try:
        parsed_expr = sympy.sympify(expr, locals=ns)
        t_sym, y_sym = sympy.Symbol("t"), sympy.Symbol("y")
        f_lambdified = sympy.lambdify((t_sym, y_sym), parsed_expr, "math")

        def f(tv, yv):
            try:
                val = f_lambdified(tv, yv)
                if isinstance(val, complex) and val.imag == 0:
                    val = val.real
                return float(val)
            except Exception:
                return float("nan")

        return f
    except Exception as e:
        raise ValueError(f"Expresión matemática f(t,y) inválida: {e}")


# ── Number/cell parsing ────────────────────────────────────────────────────────
def parse_number_cell(s: str) -> float:
    """Convert a cell string to float safely using sympy."""
    import sympy
    t = s.strip()
    if not t:
        raise ValueError("Celda vacía.")
    try:
        return float(t)
    except ValueError:
        pass

    ns = _get_sympy_ns()
    expr = norm(t).replace("ln(", "log(")
    try:
        parsed = sympy.sympify(expr, locals=ns)
        fv = float(parsed.evalf())
        if not math.isfinite(fv):
            raise ValueError(f"Valor no finito: {s!r}")
        return fv
    except Exception as e:
        raise ValueError(f"Expresión numérica inválida '{s}': {e}")


# ── LaTeX formatting helpers ───────────────────────────────────────────────────
def _latex_float(x: float, ndec: int) -> str:
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return r"\text{—}"
    return f"{float(x):.{ndec}f}"


def _latex_bound(x: float, ndec: int) -> str:
    """Format a bound (a or b) in LaTeX without trailing decimals for integers."""
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "?"
    xf = float(x)
    if abs(xf - round(xf)) < 1e-14 * max(1.0, abs(xf)):
        return str(int(round(xf)))
    return f"{xf:.{ndec}f}"


def _fmt_iter_df(df, decimals: int = 6):
    """Format float columns of an iteration table to `decimals` decimal places."""
    fmt = {}
    for col in df.columns:
        if hasattr(df[col], "dtype") and str(df[col].dtype).startswith("float"):
            fmt[col] = f"{{:.{decimals}f}}"
    return df.style.format(fmt, na_rep="—")


def expr_integrando_latex(expr_py: str) -> str:
    """Convert a Python f(x) expression to LaTeX (integrand)."""
    try:
        from sympy import (
            Abs, E, cos, exp, latex, log, pi, root, sin, sqrt, sympify, tan,
        )

        s = norm(expr_py).replace("ln(", "log(")
        loc = {
            "pi": pi, "E": E, "e": E, "exp": exp, "log": log,
            "sin": sin, "cos": cos, "tan": tan, "sqrt": sqrt,
            "abs": Abs, "cbrt": lambda z: root(z, 3),
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


# ── Shared Streamlit rendering helpers ─────────────────────────────────────────
def _formulas_panel(algoritmo: str):
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
        elif algoritmo == "Ecuaciones Diferenciales (EDO)":
            st.markdown("**Método de Euler (Taylor orden 1)**")
            st.latex(r"y_{n+1} = y_n + h f(t_n, y_n)")
            st.markdown("**Runge-Kutta de 2do orden (Punto Medio)**")
            st.latex(r"k_1 = f(t_n, y_n)")
            st.latex(r"k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)")
            st.latex(r"y_{n+1} = y_n + h k_2")
            st.markdown("**Runge-Kutta Clásico (4to orden)**")
            st.latex(r"k_1 = f(t_n, y_n)")
            st.latex(r"k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)")
            st.latex(r"k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_2)")
            st.latex(r"k_4 = f(t_n + h, y_n + h k_3)")
            st.latex(r"y_{n+1} = y_n + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)")
            st.caption("Euler tiene error global $\\mathcal{O}(h)$. RK4 tiene error $\\mathcal{O}(h^4)$.")
        elif algoritmo == "Simulación SIR (Epidemia)":
            st.markdown("**Modelo SIR**")
            st.latex(r"\frac{dS}{dt} = -\frac{\beta \cdot S \cdot I}{N}")
            st.latex(r"\frac{dI}{dt} = \frac{\beta \cdot S \cdot I}{N} - \gamma \cdot I")
            st.latex(r"\frac{dR}{dt} = \gamma \cdot I")
            st.markdown("**Número Reproductivo**")
            st.latex(r"R_0 = \frac{\beta}{\gamma}")
            st.caption("$R_0 > 1$ → epidemia crece. $R_0 < 1$ → se extingue.")
            st.markdown("**Inmunidad de Rebaño**")
            st.latex(r"\text{Umbral} = 1 - \frac{1}{R_0}")
            st.caption("Porcentaje de la población que debe ser inmune para frenar la propagación.")
        elif algoritmo == "Lanzamiento Cohete 3D":
            st.markdown("**Modelo Físico (2D/3D)**")
            st.latex(r"\frac{dx}{dt} = v_x")
            st.latex(r"\frac{dy}{dt} = v_y")
            st.latex(r"\frac{dv_x}{dt} = \frac{T \cos(\theta)}{m} - \frac{D \cdot v_x}{m \cdot v}")
            st.latex(r"\frac{dv_y}{dt} = \frac{T \sin(\theta)}{m} - \frac{D \cdot v_y}{m \cdot v} - g(y)")
            st.latex(r"\frac{dm}{dt} = -\dot{m}")
            st.markdown("**Fuerzas**")
            st.latex(r"D = \frac{1}{2} \rho(y) v^2 C_d A")
            st.latex(r"g(y) = g_0 \left(\frac{R_E}{R_E+y}\right)^2")
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
        st.dataframe(_fmt_iter_df(df_h), hide_index=True, use_container_width=True)

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
                    e_k1  = errores_positivos[k - 1]
                    e_k2  = errores_positivos[k - 2]
                    if e_k1 > 0 and e_k2 > 0 and e_k > 0:
                        denom = math.log(abs(e_k1 / e_k2))
                        if abs(denom) > 1e-15:
                            p = math.log(abs(e_k / e_k1)) / denom
                            if math.isfinite(p) and 0 < p < 10:
                                p_vals.append(p)
                if p_vals:
                    p_est = sum(p_vals[-3:]) / len(p_vals[-3:])
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
