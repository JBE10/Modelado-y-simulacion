import math

import altair as alt
import pandas as pd
import streamlit as st

from biseccion import biseccion
from punto_fijo import punto_fijo


ALLOWED_MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
}


def normalizar_expresion(expr):
    # Usuarios suelen escribir x^2; en Python debe ser x**2.
    return expr.replace("^", "**").strip()


def crear_funcion(expr):
    expr = normalizar_expresion(expr)

    def f(x):
        contexto = {"__builtins__": {}}
        variables = {"x": x, **ALLOWED_MATH_FUNCS}
        return eval(expr, contexto, variables)

    return f


st.set_page_config(page_title="Dashboard de Algoritmos Numericos", layout="wide")

st.title("Dashboard de Algoritmos Numericos")
st.caption("Interfaz local para ejecutar metodos numericos y visualizar iteraciones.")

with st.sidebar:
    st.header("Algoritmos")
    algoritmo = st.selectbox("Selecciona un algoritmo", ["Biseccion", "Punto Fijo"])
    st.info("Iras agregando mas algoritmos aqui cuando quieras.")

if algoritmo == "Biseccion":
    st.subheader("Metodo de Biseccion")

    with st.form("form_biseccion"):
        col1, col2 = st.columns(2)

        with col1:
            expr = st.text_input("Funcion f(x)", value="x**3 - x - 2")
            a = st.number_input("Extremo izquierdo a", value=1.0, format="%.10f")
            b = st.number_input("Extremo derecho b", value=2.0, format="%.10f")

        with col2:
            tol = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f")
            max_iter = st.number_input("Maximo de iteraciones", value=100, min_value=1, step=1)
            mostrar_resumen = st.checkbox("Mostrar resumen rapido", value=True)

        ejecutar = st.form_submit_button("Calcular", type="primary")

    if ejecutar or "resultado_biseccion" in st.session_state:
        try:
            if ejecutar:
                f = crear_funcion(expr)
                raiz, iters, historial = biseccion(
                    f,
                    a,
                    b,
                    tol=tol,
                    max_iter=int(max_iter),
                    devolver_historial=True,
                )
                st.session_state["resultado_biseccion"] = {
                    "expr": expr,
                    "a": a,
                    "b": b,
                    "tol": tol,
                    "max_iter": int(max_iter),
                    "mostrar_resumen": mostrar_resumen,
                    "raiz": raiz,
                    "iters": iters,
                    "historial": historial,
                }

            resultado = st.session_state["resultado_biseccion"]
            f = crear_funcion(resultado["expr"])
            raiz = resultado["raiz"]
            iters = resultado["iters"]
            historial = resultado["historial"]
            mostrar_resumen = resultado["mostrar_resumen"]
            a = resultado["a"]
            b = resultado["b"]

            st.success("Calculo completado correctamente.")

            if mostrar_resumen:
                c1, c2, c3 = st.columns(3)
                c1.metric("Raiz aproximada", f"{raiz:.12f}")
                c2.metric("Iteraciones", f"{iters}")
                c3.metric("|f(raiz)|", f"{abs(f(raiz)):.4e}")

            df_historial = pd.DataFrame(historial)
            if not df_historial.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(
                    df_historial,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("### Grafica de convergencia")
                df_conv = df_historial[["iter", "c", "error"]].copy()
                chart_c = (
                    alt.Chart(df_conv)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("iter:Q", title="Iteracion"),
                        y=alt.Y("c:Q", title="Valor de c"),
                        tooltip=["iter", "c", "error"],
                    )
                    .properties(height=280)
                )
                chart_error = (
                    alt.Chart(df_conv)
                    .mark_line(point=True, color="#d62728")
                    .encode(
                        x=alt.X("iter:Q", title="Iteracion"),
                        y=alt.Y("error:Q", title="Error"),
                        tooltip=["iter", "error"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_c, width="stretch")
                st.altair_chart(chart_error, width="stretch")
            else:
                st.warning("No se genero historial de iteraciones.")

            st.markdown("### Grafica de f(x) en el intervalo")
            x_min = min(a, b)
            x_max = max(a, b)
            margen = max((x_max - x_min) * 0.2, 1e-3)
            n_puntos = 250
            paso = (x_max - x_min + 2 * margen) / (n_puntos - 1)
            x_values = [x_min - margen + i * paso for i in range(n_puntos)]

            datos = []
            for x in x_values:
                try:
                    y = f(x)
                    if isinstance(y, complex):
                        continue
                    datos.append({"x": x, "fx": float(y)})
                except Exception:
                    continue

            if datos:
                df_fx = pd.DataFrame(datos)
                curva = (
                    alt.Chart(df_fx)
                    .mark_line()
                    .encode(
                        x=alt.X("x:Q", title="x"),
                        y=alt.Y("fx:Q", title="f(x)"),
                        tooltip=["x", "fx"],
                    )
                    .properties(height=320)
                )
                eje_x = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
                raiz_point = alt.Chart(
                    pd.DataFrame({"x": [raiz], "fx": [f(raiz)]})
                ).mark_point(color="red", size=100).encode(x="x:Q", y="fx:Q", tooltip=["x", "fx"])
                st.altair_chart(curva + eje_x + raiz_point, width="stretch")
            else:
                st.warning("No fue posible construir la grafica de f(x) para ese intervalo.")

        except Exception as error:
            st.error(f"No se pudo ejecutar el metodo: {error}")

if algoritmo == "Punto Fijo":
    st.subheader("Metodo de Punto Fijo")

    with st.form("form_punto_fijo"):
        col1, col2 = st.columns(2)

        with col1:
            expr_g = st.text_input("Funcion iterativa g(x)", value="(x + 2)**(1/3)")
            expr_f = st.text_input("Funcion original f(x) (opcional, para mostrar residuo y grafica)", value="x**3 - x - 2")
            x0 = st.number_input("Valor inicial x0", value=1.5, format="%.10f")

        with col2:
            tol_pf = st.number_input("Tolerancia", value=1e-7, min_value=1e-15, format="%.15f", key="tol_pf")
            max_iter_pf = st.number_input("Maximo de iteraciones", value=100, min_value=1, step=1, key="max_iter_pf")
            mostrar_resumen_pf = st.checkbox("Mostrar resumen rapido", value=True, key="mostrar_resumen_pf")

        ejecutar_pf = st.form_submit_button("Calcular", type="primary")

    if ejecutar_pf or "resultado_punto_fijo" in st.session_state:
        try:
            if ejecutar_pf:
                g = crear_funcion(expr_g)
                raiz, iters, historial = punto_fijo(
                    g,
                    x0,
                    tol=tol_pf,
                    max_iter=int(max_iter_pf),
                    devolver_historial=True,
                )
                st.session_state["resultado_punto_fijo"] = {
                    "expr_g": expr_g,
                    "expr_f": expr_f,
                    "x0": x0,
                    "tol": tol_pf,
                    "max_iter": int(max_iter_pf),
                    "mostrar_resumen": mostrar_resumen_pf,
                    "raiz": raiz,
                    "iters": iters,
                    "historial": historial,
                }

            resultado_pf = st.session_state["resultado_punto_fijo"]
            g = crear_funcion(resultado_pf["expr_g"])
            expr_f_guardada = resultado_pf["expr_f"].strip()
            f = crear_funcion(expr_f_guardada) if expr_f_guardada else None
            raiz = resultado_pf["raiz"]
            iters = resultado_pf["iters"]
            historial = resultado_pf["historial"]
            mostrar_resumen_pf = resultado_pf["mostrar_resumen"]
            x0 = resultado_pf["x0"]

            st.success("Calculo completado correctamente.")

            if mostrar_resumen_pf:
                c1, c2, c3 = st.columns(3)
                c1.metric("Punto fijo aproximado", f"{raiz:.12f}")
                c2.metric("Iteraciones", f"{iters}")
                if f is not None:
                    c3.metric("|f(raiz)|", f"{abs(f(raiz)):.4e}")
                else:
                    c3.metric("|g(raiz)-raiz|", f"{abs(g(raiz) - raiz):.4e}")

            df_historial_pf = pd.DataFrame(historial)
            if not df_historial_pf.empty:
                st.markdown("### Tabla de iteraciones")
                st.dataframe(
                    df_historial_pf,
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("### Grafica de convergencia")
                df_conv_pf = df_historial_pf[["iter", "x_n+1", "error"]].copy()
                chart_x = (
                    alt.Chart(df_conv_pf)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("iter:Q", title="Iteracion"),
                        y=alt.Y("x_n+1:Q", title="Valor de x"),
                        tooltip=["iter", "x_n+1", "error"],
                    )
                    .properties(height=280)
                )
                chart_error_pf = (
                    alt.Chart(df_conv_pf)
                    .mark_line(point=True, color="#d62728")
                    .encode(
                        x=alt.X("iter:Q", title="Iteracion"),
                        y=alt.Y("error:Q", title="Error"),
                        tooltip=["iter", "error"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(chart_x, width="stretch")
                st.altair_chart(chart_error_pf, width="stretch")
            else:
                st.warning("No se genero historial de iteraciones.")

            if f is not None:
                st.markdown("### Grafica de f(x) alrededor del punto fijo")
                x_min = min(x0, raiz) - 1.0
                x_max = max(x0, raiz) + 1.0
                n_puntos = 250
                paso = (x_max - x_min) / (n_puntos - 1)
                x_values = [x_min + i * paso for i in range(n_puntos)]

                datos = []
                for x in x_values:
                    try:
                        y = f(x)
                        if isinstance(y, complex):
                            continue
                        datos.append({"x": x, "fx": float(y)})
                    except Exception:
                        continue

                if datos:
                    df_fx = pd.DataFrame(datos)
                    curva = (
                        alt.Chart(df_fx)
                        .mark_line()
                        .encode(
                            x=alt.X("x:Q", title="x"),
                            y=alt.Y("fx:Q", title="f(x)"),
                            tooltip=["x", "fx"],
                        )
                        .properties(height=320)
                    )
                    eje_x = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
                    raiz_point = alt.Chart(
                        pd.DataFrame({"x": [raiz], "fx": [f(raiz)]})
                    ).mark_point(color="red", size=100).encode(x="x:Q", y="fx:Q", tooltip=["x", "fx"])
                    st.altair_chart(curva + eje_x + raiz_point, width="stretch")
                else:
                    st.warning("No fue posible construir la grafica de f(x) con esa expresion.")

        except Exception as error:
            st.error(f"No se pudo ejecutar el metodo: {error}")

st.markdown("---")
st.caption("Siguiente paso sugerido: agregar Newton-Raphson, Secante y Falsa Posicion.")
