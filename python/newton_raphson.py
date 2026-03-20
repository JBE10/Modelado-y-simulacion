import math


def _derivada_numerica(f, x):
    h = 1e-6 * (1.0 + abs(x))
    return (f(x + h) - f(x - h)) / (2.0 * h)


def newton_raphson(
    f,
    x0,
    tol=1e-7,
    max_iter=100,
    df=None,
    mostrar_tabla=False,
    devolver_historial=False,
):
    """
    Encuentra una raiz de f(x)=0 usando Newton-Raphson.

    Si no se provee df, se aproxima con diferencia centrada.
    """
    if tol <= 0:
        raise ValueError("La tolerancia debe ser mayor a 0.")
    if max_iter < 1:
        raise ValueError("max_iter debe ser >= 1.")

    historial = []
    xn = float(x0)
    derivada_numerica = df is None

    if mostrar_tabla:
        print(f"{'Iter':>4} | {'x_n':>14} | {'f(x_n)':>12} | {'df(x_n)':>12} | {'x_n+1':>14} | {'error':>12}")
        print("-" * 86)

    for i in range(1, max_iter + 1):
        try:
            fx = float(f(xn))
        except Exception as error:
            return _resultado(
                xn,
                i - 1,
                historial,
                False,
                f"Se detuvo en iter {i}: no se pudo evaluar f(x_n). Error: {error}",
                derivada_numerica,
                True,
                devolver_historial,
            )

        if not math.isfinite(fx):
            return _resultado(
                xn,
                i - 1,
                historial,
                False,
                f"Se detuvo en iter {i}: f(x_n) dio NaN/Inf.",
                derivada_numerica,
                True,
                devolver_historial,
            )

        try:
            dfx = float(_derivada_numerica(f, xn) if derivada_numerica else df(xn))
        except Exception as error:
            return _resultado(
                xn,
                i - 1,
                historial,
                False,
                f"Se detuvo en iter {i}: no se pudo evaluar f'(x_n). Error: {error}",
                derivada_numerica,
                True,
                devolver_historial,
            )

        if not math.isfinite(dfx):
            return _resultado(
                xn,
                i - 1,
                historial,
                False,
                f"Se detuvo en iter {i}: f'(x_n) dio NaN/Inf.",
                derivada_numerica,
                True,
                devolver_historial,
            )

        if abs(dfx) < 1e-14:
            return _resultado(
                xn,
                i - 1,
                historial,
                False,
                f"Se detuvo en iter {i}: derivada casi nula (|f'(x_n)| < 1e-14).",
                derivada_numerica,
                True,
                devolver_historial,
            )

        x_next = xn - fx / dfx
        error = abs(x_next - xn)
        fila = {"iter": i, "x_n": xn, "f(x_n)": fx, "f'(x_n)": dfx, "x_n+1": x_next, "error": error}
        historial.append(fila)

        if mostrar_tabla:
            print(f"{i:4d} | {xn:14.8f} | {fx:12.4e} | {dfx:12.4e} | {x_next:14.8f} | {error:12.4e}")

        if error < tol or abs(fx) < tol:
            return _resultado(
                x_next,
                i,
                historial,
                True,
                f"Convergio en iter {i}: se cumplio |x_n+1 - x_n| < tol o |f(x_n)| < tol.",
                derivada_numerica,
                False,
                devolver_historial,
            )

        xn = x_next

    return _resultado(
        xn,
        max_iter,
        historial,
        False,
        f"No convergio: se alcanzo max_iter={max_iter}. Revisa x0 o la derivada usada.",
        derivada_numerica,
        False,
        devolver_historial,
    )


def _resultado(raiz, iteraciones, historial, convergio, justificacion, derivada_numerica, terminado_por_error, devolver_historial):
    out = {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "historial": historial,
        "convergio": convergio,
        "justificacion": justificacion,
        "derivada_numerica": derivada_numerica,
        "terminado_por_error": terminado_por_error,
    }
    if devolver_historial:
        return out
    return {k: v for k, v in out.items() if k != "historial"}
