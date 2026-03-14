import math


def punto_fijo(g, x0, tol=1e-7, max_iter=100, mostrar_tabla=False, devolver_historial=False):
    """
    Encuentra una aproximacion a una solucion de x = g(x) usando iteracion de punto fijo.

    Parametros:
    - g: funcion iterativa
    - x0: valor inicial
    - tol: tolerancia para el error |x_{n+1} - x_n|
    - max_iter: maximo numero de iteraciones
    - mostrar_tabla: si es True, imprime una tabla con cada iteracion
    - devolver_historial: si es True, retorna tambien una lista con el detalle de iteraciones

    Retorna:
    - (raiz_aproximada, numero_iteraciones)
    - (raiz_aproximada, numero_iteraciones, historial) si devolver_historial=True
    """
    if mostrar_tabla:
        print(f"{'Iter':>4} | {'x_n':>14} | {'x_n+1':>14} | {'error':>12}")
        print("-" * 56)

    historial = []
    xn = float(x0)

    for i in range(1, max_iter + 1):
        x_next = g(xn)
        if isinstance(x_next, complex):
            raise ValueError("g(x) devolvio un valor complejo. Revisa la expresion o el valor inicial.")
        if not math.isfinite(float(x_next)):
            raise ValueError("g(x) devolvio NaN o infinito. Revisa la expresion o el valor inicial.")

        error = abs(float(x_next) - xn)

        historial.append(
            {
                "iter": i,
                "x_n": xn,
                "x_n+1": x_next,
                "error": error,
            }
        )

        if mostrar_tabla:
            print(f"{i:4d} | {xn:14.8f} | {x_next:14.8f} | {error:12.4e}")

        if error < tol:
            if devolver_historial:
                return float(x_next), i, historial
            return float(x_next), i

        xn = float(x_next)

    if devolver_historial:
        return xn, max_iter, historial
    return xn, max_iter
