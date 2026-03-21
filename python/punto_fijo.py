import math


def punto_fijo(g, x0, tol=1e-4, max_iter=20, f=None):
    historial = []
    xn = float(x0)

    for i in range(1, max_iter + 1):
        try:
            x_next = float(g(xn))
        except Exception as e:
            return _res(xn, i - 1, historial, False, f"No se pudo evaluar g(x_n) en iter {i}. {e}")

        if not math.isfinite(x_next):
            return _res(xn, i - 1, historial, False, f"g(x_n) no es finito en iter {i}: {x_next}.")

        error = abs(x_next - xn)
        fila = {"iter": i, "x_n": xn, "x_n+1": x_next, "error": error}
        if f is not None:
            try:
                fila["|f(x_n+1)|"] = abs(float(f(x_next)))
            except Exception:
                fila["|f(x_n+1)|"] = None
        historial.append(fila)

        if error < tol:
            return _res(x_next, i, historial, True, f"Convergio en {i} iteraciones.")

        xn = x_next

    return _res(xn, max_iter, historial, False,
                f"No convergio en {max_iter} iteraciones.")


def _res(raiz, iteraciones, historial, convergio, justificacion):
    return {"raiz": raiz, "iteraciones": iteraciones, "historial": historial,
            "convergio": convergio, "justificacion": justificacion}
