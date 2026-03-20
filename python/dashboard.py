import math

import altair as alt
import pandas as pd
import streamlit as st

from biseccion import biseccion
from newton_raphson import newton_raphson
from punto_fijo import punto_fijo


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


st.set_page_config(page_title="Dashboard de Algoritmos Numericos", layout="wide")

st.title("Dashboard de Algoritmos Numericos")
st.caption("Interfaz local para ejecutar metodos numericos y visualizar iteraciones.")
st.markdown(
    "**Funciones soportadas:** `sin`, `cos`, `tan`, `exp`, `log`, `ln`, `sqrt`, `cbrt`, `abs` — "
    "Constantes: `pi`, `e` — Potencias: `^` o `**`"
)

with st.sidebar:
    st.header("Algoritmos")
    algoritmo = st.selectbox("Selecciona un algoritmo", ["Biseccion", "Punto Fijo", "Newton-Raphson"])


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
                x_min = min(a_res, b_res)
                x_max = max(a_res, b_res)
                margen = max((x_max - x_min) * 0.2, 1e-3)
                n_puntos = 250
                paso = (x_max - x_min + 2 * margen) / (n_puntos - 1)
                datos = []
                for idx in range(n_puntos):
                    xv = x_min - margen + idx * paso
                    try:
                        yv = f(xv)
                        if not isinstance(yv, complex) and math.isfinite(yv):
                            datos.append({"x": xv, "fx": float(yv)})
                    except Exception:
                        pass
                if datos:
                    df_fx = pd.DataFrame(datos)
                    curva = alt.Chart(df_fx).mark_line().encode(x="x:Q", y="fx:Q", tooltip=["x", "fx"]).properties(height=320)
                    eje = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
                    pt = alt.Chart(pd.DataFrame({"x": [raiz], "fx": [f(raiz)]})).mark_point(color="red", size=100).encode(x="x:Q", y="fx:Q")
                    st.altair_chart(curva + eje + pt, width="stretch")

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
                x0_res = res["x0"]
                x_min = min(x0_res, raiz) - 1.0
                x_max = max(x0_res, raiz) + 1.0
                n_puntos = 250
                paso = (x_max - x_min) / (n_puntos - 1)
                datos = []
                for idx in range(n_puntos):
                    xv = x_min + idx * paso
                    try:
                        yv = f_fn(xv)
                        if not isinstance(yv, complex) and math.isfinite(yv):
                            datos.append({"x": xv, "fx": float(yv)})
                    except Exception:
                        pass
                if datos:
                    df_fx = pd.DataFrame(datos)
                    curva = alt.Chart(df_fx).mark_line().encode(x="x:Q", y="fx:Q", tooltip=["x", "fx"]).properties(height=320)
                    eje = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
                    pt = alt.Chart(pd.DataFrame({"x": [raiz], "fx": [f_fn(raiz)]})).mark_point(color="red", size=100).encode(x="x:Q", y="fx:Q")
                    st.altair_chart(curva + eje + pt, width="stretch")

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
                x0_res = res["x0"]
                x_min = min(x0_res, raiz) - 1.0
                x_max = max(x0_res, raiz) + 1.0
                n_puntos = 250
                paso = (x_max - x_min) / (n_puntos - 1)
                datos = []
                for idx in range(n_puntos):
                    xv = x_min + idx * paso
                    try:
                        yv = f_fn(xv)
                        if not isinstance(yv, complex) and math.isfinite(yv):
                            datos.append({"x": xv, "fx": float(yv)})
                    except Exception:
                        pass
                if datos:
                    df_fx = pd.DataFrame(datos)
                    curva = alt.Chart(df_fx).mark_line().encode(x="x:Q", y="fx:Q", tooltip=["x", "fx"]).properties(height=320)
                    eje = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
                    pt = alt.Chart(pd.DataFrame({"x": [raiz], "fx": [f_fn(raiz)]})).mark_point(color="red", size=100).encode(x="x:Q", y="fx:Q")
                    st.altair_chart(curva + eje + pt, width="stretch")

        except Exception as error:
            st.error(f"Error: {error}")

st.markdown("---")
st.caption("Proximo paso: agregar Secante y Falsa Posicion.")
