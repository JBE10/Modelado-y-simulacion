import math

import altair as alt
import pandas as pd
import streamlit as st

from biseccion import biseccion
from newton_raphson import newton_raphson
from punto_fijo import punto_fijo
from secante import secante


def cbrt_safe(x):
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


ALLOWED_MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sqrt": math.sqrt,
    "cbrt": cbrt_safe,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
}


def normalizar_expresion(expr):
    return expr.replace("^", "**").strip()


def crear_funcion(expr):
    expr = normalizar_expresion(expr)

    def f(x):
        return eval(expr, {"__builtins__": {}}, {"x": x, **ALLOWED_MATH_FUNCS})

    return f


def derivada_numerica(f, x):
    h = 1e-6 * (1.0 + abs(x))
    return (f(x + h) - f(x - h)) / (2.0 * h)


def derivada_simbolica(expr_str):
    """Devuelve f'(x) y g(x) como strings simbolicos usando sympy."""
    import sympy
    x = sympy.Symbol("x")
    expr_str_norm = normalizar_expresion(expr_str)
    replacements = {
        "ln(": "log(",
        "cbrt(": "(", 
    }
    expr_sym = expr_str_norm
    for old, new in replacements.items():
        expr_sym = expr_sym.replace(old, new)
    if "cbrt" in expr_str_norm:
        expr_sym = expr_str_norm.replace("cbrt(", "Pow(").replace(")", ", Rational(1,3))", 1)

    try:
        f_sym = sympy.sympify(expr_sym, locals={"x": x})
        fp_sym = sympy.diff(f_sym, x)
        fp_simplified = sympy.simplify(fp_sym)
        g_sym = sympy.simplify(x - f_sym / fp_simplified)
        return str(fp_simplified), str(g_sym)
    except Exception:
        return None, None


def graficar_fx(f, raiz, x_min, x_max, expr_label="f(x)", n_puntos=400):
    """Genera una grafica estilo GeoGebra: ejes, grilla, curva f(x), punto raiz."""
    margen = max((x_max - x_min) * 0.3, 0.5)
    xa = x_min - margen
    xb = x_max + margen
    paso = (xb - xa) / (n_puntos - 1)

    datos = []
    for idx in range(n_puntos):
        xv = xa + idx * paso
        try:
            yv = f(xv)
            if not isinstance(yv, complex) and math.isfinite(yv):
                datos.append({"x": round(xv, 10), "f(x)": round(float(yv), 10)})
        except Exception:
            pass

    if not datos:
        return

    df_fx = pd.DataFrame(datos)
    y_vals = [d["f(x)"] for d in datos]
    y_lo = min(y_vals)
    y_hi = max(y_vals)
    y_pad = max((y_hi - y_lo) * 0.1, 0.5)

    curva = (
        alt.Chart(df_fx)
        .mark_line(strokeWidth=2.5, color="#2563eb")
        .encode(
            x=alt.X("x:Q", title="x", scale=alt.Scale(domain=[xa, xb])),
            y=alt.Y("f(x):Q", title=expr_label, scale=alt.Scale(domain=[y_lo - y_pad, y_hi + y_pad])),
            tooltip=[alt.Tooltip("x:Q", format=".6f"), alt.Tooltip("f(x):Q", format=".6e")],
        )
        .properties(height=400)
    )

    eje_x = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#6b7280", strokeWidth=1.5, strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    eje_y = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#6b7280", strokeWidth=1.5, strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    try:
        fr = float(f(raiz))
    except Exception:
        fr = 0.0
    df_raiz = pd.DataFrame({
        "x": [raiz],
        "f(x)": [fr],
        "label": [f"raiz ≈ {raiz:.8f}"],
    })

    punto = (
        alt.Chart(df_raiz)
        .mark_point(size=180, color="#ef4444", filled=True, stroke="#ffffff", strokeWidth=1.5)
        .encode(
            x="x:Q",
            y="f(x):Q",
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("x:Q", format=".10f"), alt.Tooltip("f(x):Q", format=".4e")],
        )
    )

    etiqueta = (
        alt.Chart(df_raiz)
        .mark_text(align="left", dx=12, dy=-12, fontSize=13, fontWeight="bold", color="#ef4444")
        .encode(x="x:Q", y="f(x):Q", text="label:N")
    )

    chart = (curva + eje_x + eje_y + punto + etiqueta).configure_axis(
        grid=True,
        gridColor="#e5e7eb",
        gridOpacity=0.4,
    )

    st.altair_chart(chart, use_container_width=True)


st.set_page_config(page_title="Dashboard de Algoritmos Numericos", layout="wide")

st.title("Dashboard de Algoritmos Numericos")
st.caption("Interfaz local para ejecutar metodos numericos y visualizar iteraciones.")
st.markdown(
    "**Funciones soportadas:** `sin`, `cos`, `tan`, `exp`, `log`, `ln`, `sqrt`, `cbrt`, `abs` — "
    "Constantes: `pi`, `e` — Potencias: `^` o `**`"
)

with st.sidebar:
    st.header("Algoritmos")
    algoritmo = st.selectbox("Selecciona un algoritmo", ["Biseccion", "Punto Fijo", "Newton-Raphson", "Secante"])


# ---------------------------------------------------------------------------
#  BISECCION
# ---------------------------------------------------------------------------
if algoritmo == "Biseccion":
    st.subheader("Metodo de Biseccion")

    with st.form("form_biseccion"):
        col1, col2 = st.columns(2)

        with col1:
            expr = st.text_input("f(x)", value="x**3 - x - 2")
            a = st.number_input("Extremo izquierdo a", value=1.0, format="%.10f")
            b = st.number_input("Extremo derecho b", value=2.0, format="%.10f")

        with col2:
            tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f")
            max_iter = st.number_input("Maximo de iteraciones", value=100, min_value=1, step=1)

        ejecutar = st.form_submit_button("Calcular", type="primary")

    if ejecutar or "resultado_biseccion" in st.session_state:
        try:
            if ejecutar:
                f = crear_funcion(expr)
                raiz, iters, historial = biseccion(
                    f, a, b, tol=tol, max_iter=int(max_iter), devolver_historial=True,
                )
                convergio = iters < int(max_iter) or (historial and abs(historial[-1]["f(c)"]) < tol)

                justif = ""
                if convergio:
                    justif = f"Convergio en iter {iters}: se cumplio |f(c)| < tol o error < tol."
                else:
                    ancho = abs(b - a)
                    requerido = math.ceil(math.log2(ancho / tol)) if tol > 0 and ancho > 0 else max_iter
                    justif = (
                        f"No convergio: se alcanzo max_iter={int(max_iter)}. "
                        f"Se estiman ~{requerido} iteraciones para esa tolerancia."
                    )

                st.session_state["resultado_biseccion"] = {
                    "expr": expr, "a": a, "b": b, "tol": tol,
                    "max_iter": int(max_iter),
                    "raiz": raiz, "iters": iters, "historial": historial,
                    "convergio": convergio, "justificacion": justif,
                }

            res = st.session_state["resultado_biseccion"]
            f = crear_funcion(res["expr"])
            raiz = res["raiz"]
            iters = res["iters"]
            historial = res["historial"]
            a_res = res["a"]
            b_res = res["b"]

            if res["convergio"]:
                st.success(res["justificacion"])
            else:
                st.warning(res["justificacion"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Raiz aproximada", f"{raiz:.12f}")
            c2.metric("Iteraciones", f"{iters}")
            c3.metric("|f(raiz)|", f"{abs(f(raiz)):.4e}")

            expr_norm = normalizar_expresion(res["expr"])
            fp_str, g_str = derivada_simbolica(res["expr"])
            if fp_str:
                st.markdown("### Derivada y g(x)")
                st.code(
                    f"f(x)  = {expr_norm}\n"
                    f"f'(x) = {fp_str}\n"
                    f"g(x)  = x - f(x)/f'(x) = {g_str}"
                )

            df_hist = pd.DataFrame(historial)
            if not df_hist.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(df_hist, width="stretch", hide_index=True)

                st.markdown("### Graficas")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    chart_c = (
                        alt.Chart(df_hist).mark_line(point=True)
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("c:Q", title="c"), tooltip=["iter", "c", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_c, width="stretch")
                with col_g2:
                    chart_err = (
                        alt.Chart(df_hist).mark_line(point=True, color="#d62728")
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("error:Q", title="Error"), tooltip=["iter", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_err, width="stretch")

                st.markdown("### Grafica de f(x)")
                graficar_fx(f, raiz, min(a_res, b_res), max(a_res, b_res), expr_label=res["expr"])

        except Exception as error:
            st.error(f"Error: {error}")


# ---------------------------------------------------------------------------
#  PUNTO FIJO
# ---------------------------------------------------------------------------
if algoritmo == "Punto Fijo":
    st.subheader("Metodo de Punto Fijo")

    with st.form("form_punto_fijo"):
        col1, col2 = st.columns(2)

        with col1:
            expr_f = st.text_input("f(x)", value="x**3 - x - 2")
            expr_g = st.text_input("g(x)  (opcional: si vacio se usa g(x)=x-f(x)/f'(x))", value="")
            x0 = st.number_input("Valor inicial x0", value=1.5, format="%.10f")

        with col2:
            tol_pf = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_pf")
            max_iter_pf = st.number_input("Maximo de iteraciones", value=100, min_value=1, step=1, key="max_iter_pf")
            usar_aitken = st.checkbox("Aplicar aceleracion de Aitken", value=False, key="aitken_pf")

        ejecutar_pf = st.form_submit_button("Calcular", type="primary")

    if ejecutar_pf or "resultado_punto_fijo" in st.session_state:
        try:
            if ejecutar_pf:
                if not expr_f.strip():
                    st.error("Debes ingresar f(x).")
                    st.stop()

                g_auto = False
                usa_derivada_numerica = False
                expr_f_norm = normalizar_expresion(expr_f)
                if not expr_g.strip():
                    expr_g_usado = f"x - ({expr_f_norm}) / f'(x)"
                    g_auto = True
                    usa_derivada_numerica = True
                else:
                    expr_g_usado = expr_g

                f_fn = crear_funcion(expr_f)
                if usa_derivada_numerica:
                    def g_fn(x):
                        dfx = derivada_numerica(f_fn, x)
                        if not math.isfinite(dfx) or abs(dfx) < 1e-14:
                            raise ValueError("f'(x) no es usable (NaN/Inf o casi 0) para construir g(x).")
                        return x - f_fn(x) / dfx
                else:
                    g_fn = crear_funcion(expr_g_usado)

                res = punto_fijo(
                    g_fn, x0, tol=tol_pf, max_iter=int(max_iter_pf),
                    f=f_fn, devolver_historial=True,
                )
                res["expr_f"] = expr_f
                res["expr_f_norm"] = expr_f_norm
                res["expr_g_usado"] = expr_g_usado
                res["g_auto"] = g_auto
                res["usa_derivada_numerica"] = usa_derivada_numerica
                res["usar_aitken"] = usar_aitken
                res["x0"] = x0
                st.session_state["resultado_punto_fijo"] = res

            res = st.session_state["resultado_punto_fijo"]
            f_fn = crear_funcion(res["expr_f"])
            raiz = res["raiz"]
            iters = res["iteraciones"]
            historial = res["historial"]

            if res.get("g_auto"):
                st.info(f"g(x) generada automaticamente: `{res['expr_g_usado']}`")
            if res.get("usa_derivada_numerica"):
                st.caption("Se estima f'(x) por diferencia centrada en cada iteracion.")
                st.code(
                    "f'(x) ≈ (f(x + h) - f(x - h)) / (2h),  h = 1e-6 * (1 + |x|)\n"
                    f"g(x) = x - ({res.get('expr_f_norm', res['expr_f'])}) / f'(x)"
                )
                try:
                    dfx0 = derivada_numerica(f_fn, res["x0"])
                    dfr = derivada_numerica(f_fn, raiz)
                    if math.isfinite(dfx0) and math.isfinite(dfr):
                        st.caption(f"Valores estimados: f'(x0) ≈ {dfx0:.6e},  f'(raiz) ≈ {dfr:.6e}")
                except Exception:
                    pass

            if res["convergio"]:
                st.success(res["justificacion"])
            else:
                st.warning(res["justificacion"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Punto fijo aproximado", f"{raiz:.12f}")
            c2.metric("Iteraciones", f"{iters}")
            try:
                c3.metric("|f(raiz)|", f"{abs(f_fn(raiz)):.4e}")
            except Exception:
                c3.metric("|f(raiz)|", "N/A")

            df_hist = pd.DataFrame(historial)

            if res.get("usar_aitken") and len(historial) >= 3:
                st.markdown("### Aceleracion de Aitken")
                st.code(
                    "x̂_n = x_n - (x_{n+1} - x_n)^2 / (x_{n+2} - 2*x_{n+1} + x_n)"
                )
                x_seq = [historial[0]["x_n"]] + [h["x_n+1"] for h in historial]
                aitken_rows = []
                for k in range(len(x_seq) - 2):
                    x0a = x_seq[k]
                    x1a = x_seq[k + 1]
                    x2a = x_seq[k + 2]
                    denom = x2a - 2.0 * x1a + x0a
                    if abs(denom) < 1e-14:
                        x_hat = None
                        err_aitken = None
                    else:
                        x_hat = x0a - (x1a - x0a) ** 2 / denom
                        err_aitken = abs(x_hat - x0a)
                    aitken_rows.append({
                        "n": k,
                        "x_n": x0a,
                        "x_{n+1}": x1a,
                        "x_{n+2}": x2a,
                        "x̂_n (Aitken)": x_hat,
                        "error_aitken": err_aitken,
                    })
                df_aitken = pd.DataFrame(aitken_rows)
                st.dataframe(df_aitken, width="stretch", hide_index=True)

                aitken_valid = [r for r in aitken_rows if r["x̂_n (Aitken)"] is not None]
                if aitken_valid:
                    best = min(aitken_valid, key=lambda r: r["error_aitken"])
                    st.caption(f"Mejor aproximacion Aitken: x̂ ≈ {best['x̂_n (Aitken)']:.12f}  (error ≈ {best['error_aitken']:.4e})")

                    df_cmp = pd.DataFrame({
                        "n": list(range(len(aitken_valid))),
                        "original": [r["x_{n+1}"] for r in aitken_valid],
                        "aitken": [r["x̂_n (Aitken)"] for r in aitken_valid],
                    })
                    df_melt = df_cmp.melt("n", var_name="serie", value_name="valor")
                    chart_cmp = (
                        alt.Chart(df_melt).mark_line(point=True)
                        .encode(
                            x=alt.X("n:Q", title="n"),
                            y=alt.Y("valor:Q", title="Aproximacion"),
                            color=alt.Color("serie:N"),
                            tooltip=["n", "serie", "valor"],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(chart_cmp, width="stretch")

            if not df_hist.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(df_hist, width="stretch", hide_index=True)

                st.markdown("### Graficas")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    chart_x = (
                        alt.Chart(df_hist).mark_line(point=True)
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("x_n+1:Q", title="x"), tooltip=["iter", "x_n+1", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_x, width="stretch")
                with col_g2:
                    chart_err = (
                        alt.Chart(df_hist).mark_line(point=True, color="#d62728")
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("error:Q", title="Error"), tooltip=["iter", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_err, width="stretch")

                st.markdown("### Grafica de f(x)")
                graficar_fx(f_fn, raiz, res["x0"] - 1.0, res["x0"] + 1.0, expr_label=res["expr_f"])

        except Exception as error:
            st.error(f"Error: {error}")


# ---------------------------------------------------------------------------
#  NEWTON-RAPHSON
# ---------------------------------------------------------------------------
if algoritmo == "Newton-Raphson":
    st.subheader("Metodo de Newton-Raphson")

    with st.form("form_newton"):
        col1, col2 = st.columns(2)

        with col1:
            expr_f_n = st.text_input("f(x)", value="x**3 - x - 2", key="f_newton")
            expr_df_n = st.text_input(
                "f'(x) (opcional, si vacio se estima numericamente)",
                value="3*x**2 - 1",
                key="df_newton",
            )
            x0_n = st.number_input("Valor inicial x0", value=1.5, format="%.10f", key="x0_newton")

        with col2:
            tol_n = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_newton")
            max_iter_n = st.number_input("Maximo de iteraciones", value=50, min_value=1, step=1, key="max_iter_newton")

        ejecutar_n = st.form_submit_button("Calcular", type="primary")

    if ejecutar_n or "resultado_newton" in st.session_state:
        try:
            if ejecutar_n:
                if not expr_f_n.strip():
                    st.error("Debes ingresar f(x).")
                    st.stop()

                f_fn = crear_funcion(expr_f_n)
                df_fn = crear_funcion(expr_df_n) if expr_df_n.strip() else None

                res = newton_raphson(
                    f_fn,
                    x0_n,
                    tol=tol_n,
                    max_iter=int(max_iter_n),
                    df=df_fn,
                    devolver_historial=True,
                )
                res["expr_f"] = expr_f_n
                res["expr_df"] = expr_df_n
                res["x0"] = x0_n
                st.session_state["resultado_newton"] = res

            res = st.session_state["resultado_newton"]
            f_fn = crear_funcion(res["expr_f"])
            raiz = res["raiz"]
            iters = res["iteraciones"]
            historial = res["historial"]

            if res.get("derivada_numerica"):
                st.info("f'(x) se estima numericamente con diferencia centrada.")

            if res["convergio"]:
                st.success(res["justificacion"])
            else:
                st.warning(res["justificacion"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Raiz aproximada", f"{raiz:.12f}")
            c2.metric("Iteraciones", f"{iters}")
            try:
                c3.metric("|f(raiz)|", f"{abs(f_fn(raiz)):.4e}")
            except Exception:
                c3.metric("|f(raiz)|", "N/A")

            expr_f_norm = normalizar_expresion(res["expr_f"])
            st.markdown("### Derivada y g(x)")
            try:
                df_x0 = derivada_numerica(f_fn, res["x0"])
                df_r = derivada_numerica(f_fn, raiz)
                st.code(
                    f"f(x)  = {expr_f_norm}\n"
                    f"f'(x) ≈ (f(x+h)-f(x-h))/(2h),  h = 1e-6*(1+|x|)\n"
                    f"g(x)  = x - f(x)/f'(x)  =  x - ({expr_f_norm}) / f'(x)\n"
                    f"\n"
                    f"f'(x0={res['x0']})  ≈ {df_x0:.6e}\n"
                    f"f'(raiz)    ≈ {df_r:.6e}"
                )
            except Exception:
                pass

            df_hist = pd.DataFrame(historial)
            if not df_hist.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(df_hist, width="stretch", hide_index=True)

                st.markdown("### Graficas")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    chart_x = (
                        alt.Chart(df_hist).mark_line(point=True)
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("x_n+1:Q", title="x"), tooltip=["iter", "x_n", "x_n+1", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_x, width="stretch")
                with col_g2:
                    chart_err = (
                        alt.Chart(df_hist).mark_line(point=True, color="#d62728")
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("error:Q", title="Error"), tooltip=["iter", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_err, width="stretch")

                st.markdown("### Grafica de f(x)")
                graficar_fx(f_fn, raiz, min(res["x0"], raiz) - 1.0, max(res["x0"], raiz) + 1.0, expr_label=res["expr_f"])

        except Exception as error:
            st.error(f"Error: {error}")

# ---------------------------------------------------------------------------
#  SECANTE
# ---------------------------------------------------------------------------
if algoritmo == "Secante":
    st.subheader("Metodo de la Secante")

    with st.form("form_secante"):
        col1, col2 = st.columns(2)

        with col1:
            expr_f_s = st.text_input("f(x)", value="x**3 - x - 2", key="f_secante")
            x0_s = st.number_input("x0", value=1.0, format="%.10f", key="x0_secante")
            x1_s = st.number_input("x1", value=2.0, format="%.10f", key="x1_secante")

        with col2:
            tol_s = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_secante")
            max_iter_s = st.number_input("Maximo de iteraciones", value=100, min_value=1, step=1, key="max_iter_secante")

        ejecutar_s = st.form_submit_button("Calcular", type="primary")

    if ejecutar_s or "resultado_secante" in st.session_state:
        try:
            if ejecutar_s:
                if not expr_f_s.strip():
                    st.error("Debes ingresar f(x).")
                    st.stop()

                f_fn = crear_funcion(expr_f_s)
                res = secante(
                    f_fn, x0_s, x1_s,
                    tol=tol_s, max_iter=int(max_iter_s),
                    devolver_historial=True,
                )
                res["expr_f"] = expr_f_s
                res["x0"] = x0_s
                res["x1"] = x1_s
                st.session_state["resultado_secante"] = res

            res = st.session_state["resultado_secante"]
            f_fn = crear_funcion(res["expr_f"])
            raiz = res["raiz"]
            iters = res["iteraciones"]
            historial = res["historial"]

            st.code(
                "x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))"
            )

            if res["convergio"]:
                st.success(res["justificacion"])
            else:
                st.warning(res["justificacion"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Raiz aproximada", f"{raiz:.12f}")
            c2.metric("Iteraciones", f"{iters}")
            try:
                c3.metric("|f(raiz)|", f"{abs(f_fn(raiz)):.4e}")
            except Exception:
                c3.metric("|f(raiz)|", "N/A")

            expr_f_norm = normalizar_expresion(res["expr_f"])
            st.markdown("### Derivada y g(x)")
            try:
                df_x0 = derivada_numerica(f_fn, res["x0"])
                df_x1 = derivada_numerica(f_fn, res["x1"])
                df_r = derivada_numerica(f_fn, raiz)
                st.code(
                    f"f(x)  = {expr_f_norm}\n"
                    f"f'(x) ≈ (f(x+h)-f(x-h))/(2h),  h = 1e-6*(1+|x|)\n"
                    f"g(x)  = x - f(x)/f'(x)  =  x - ({expr_f_norm}) / f'(x)\n"
                    f"\n"
                    f"f'(x0={res['x0']})  ≈ {df_x0:.6e}\n"
                    f"f'(x1={res['x1']})  ≈ {df_x1:.6e}\n"
                    f"f'(raiz)    ≈ {df_r:.6e}"
                )
            except Exception:
                pass

            df_hist = pd.DataFrame(historial)
            if not df_hist.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(df_hist, width="stretch", hide_index=True)

                st.markdown("### Graficas")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    chart_x = (
                        alt.Chart(df_hist).mark_line(point=True)
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("x_{n+1}:Q", title="x"), tooltip=["iter", "x_n", "x_{n+1}", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_x, width="stretch")
                with col_g2:
                    chart_err = (
                        alt.Chart(df_hist).mark_line(point=True, color="#d62728")
                        .encode(x=alt.X("iter:Q", title="Iteracion"), y=alt.Y("error:Q", title="Error"), tooltip=["iter", "error"])
                        .properties(height=280)
                    )
                    st.altair_chart(chart_err, width="stretch")

                st.markdown("### Grafica de f(x)")
                graficar_fx(f_fn, raiz, min(res["x0"], res["x1"], raiz) - 1.0, max(res["x0"], res["x1"], raiz) + 1.0, expr_label=res["expr_f"])

        except Exception as error:
            st.error(f"Error: {error}")

st.markdown("---")
st.caption("Proximo paso: agregar Falsa Posicion.")
