import math


def punto_fijo(g, x0, tol=1e-7, max_iter=100, f=None, mostrar_tabla=False, devolver_historial=False):
    """
    Encuentra una aproximacion a una solucion de x = g(x) usando iteracion de punto fijo.

    Parametros:
    - g: funcion iterativa
    - x0: valor inicial
    - tol: tolerancia para el error |x_{n+1} - x_n|
    - max_iter: maximo numero de iteraciones
    - f: funcion original f(x)=0 (para calcular |f(raiz)| como residuo)
    - mostrar_tabla: si es True, imprime una tabla con cada iteracion
    - devolver_historial: si es True, retorna tambien una lista con el detalle de iteraciones

    Retorna dict con: raiz, iteraciones, historial, convergio, justificacion
    """
    if mostrar_tabla:
        header = f"{'Iter':>4} | {'x_n':>14} | {'x_n+1':>14} | {'error':>12}"
        if f is not None:
            header += f" | {'|f(x_n+1)|':>12}"
        print(header)
        print("-" * len(header))

    historial = []
    xn = float(x0)

    for i in range(1, max_iter + 1):
        try:
            x_next = g(xn)
        except Exception as e:
            return _resultado(xn, i - 1, historial, False,
                              f"Se detuvo en iter {i}: no se pudo evaluar g(x_n) con x_n={xn:.10g}. Error: {e}")

        if isinstance(x_next, complex):
            return _resultado(xn, i - 1, historial, False,
                              f"Se detuvo en iter {i}: g(x) devolvio valor complejo con x_n={xn:.10g}.")

        x_next = float(x_next)
        if not math.isfinite(x_next):
            return _resultado(xn, i - 1, historial, False,
                              f"Se detuvo en iter {i}: g(x) devolvio NaN/Inf con x_n={xn:.10g}.")

        error = abs(x_next - xn)

        residual = None
        if f is not None:
            try:
                residual = abs(float(f(x_next)))
            except Exception:
                residual = None

        fila = {"iter": i, "x_n": xn, "x_n+1": x_next, "error": error}
        if f is not None:
            fila["|f(x_n+1)|"] = residual
        historial.append(fila)

        if mostrar_tabla:
            line = f"{i:4d} | {xn:14.8f} | {x_next:14.8f} | {error:12.4e}"
            if f is not None and residual is not None:
                line += f" | {residual:12.4e}"
            print(line)

        if error < tol:
            return _resultado(x_next, i, historial, True,
                              f"Convergio en iter {i}: se cumplio |x_n+1 - x_n| < tol.")

        xn = x_next

    justif = f"No convergio: se alcanzo max_iter={max_iter}."
    try:
        h = 1e-6 * (1.0 + abs(xn))
        gprime = abs((g(xn + h) - g(xn - h)) / (2.0 * h))
        if gprime >= 1.0:
            justif += f" |g'(x)| ≈ {gprime:.4f} >= 1 cerca de la ultima aprox (no es contractiva)."
        else:
            justif += f" |g'(x)| ≈ {gprime:.4f} < 1, probablemente faltan mas iteraciones o mejor x0."
    except Exception:
        justif += " No se pudo estimar g'(x)."

    return _resultado(xn, max_iter, historial, False, justif)


def _resultado(raiz, iteraciones, historial, convergio, justificacion):
    return {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "historial": historial,
        "convergio": convergio,
        "justificacion": justificacion,
    }
