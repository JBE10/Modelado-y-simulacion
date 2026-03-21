import math


def secante(f, x0, x1, tol=1e-7, max_iter=100):
    historial = []
    xprev, xcurr = float(x0), float(x1)
    try:
        fprev = float(f(xprev))
        fcurr = float(f(xcurr))
    except Exception as e:
        return _res(xcurr, 0, historial, False, f"No se pudo evaluar f en los puntos iniciales. {e}")

    if not math.isfinite(fprev) or not math.isfinite(fcurr):
        return _res(xcurr, 0, historial, False, "f(x0) o f(x1) no es finito.")
    if abs(fprev) < tol:
        return _res(xprev, 0, historial, True, f"Convergio en 0 iter: |f(x0)| = {abs(fprev):.4e} < tol.")
    if abs(fcurr) < tol:
        return _res(xcurr, 0, historial, True, f"Convergio en 0 iter: |f(x1)| = {abs(fcurr):.4e} < tol.")

    for i in range(1, max_iter + 1):
        denom = fcurr - fprev
        if abs(denom) < 1e-14:
            return _res(xcurr, i - 1, historial, False, f"Denominador casi nulo en iter {i}.")

        x_next = xcurr - fcurr * (xcurr - xprev) / denom
        if not math.isfinite(x_next):
            return _res(xcurr, i - 1, historial, False, f"x_(n+1) no es finito en iter {i}.")

        try:
            fnext = float(f(x_next))
        except Exception as e:
            return _res(xcurr, i - 1, historial, False, f"No se pudo evaluar f(x_(n+1)) en iter {i}. {e}")

        error = abs(x_next - xcurr)
        historial.append({"iter": i, "x_{n-1}": xprev, "x_n": xcurr,
                          "f(x_n)": fcurr, "x_{n+1}": x_next, "error": error})

        if not math.isfinite(fnext):
            return _res(x_next, i, historial, False, f"f(x_(n+1)) no es finito en iter {i}: {fnext}.")
        if abs(fnext) < tol or error < tol:
            return _res(x_next, i, historial, True, f"Convergio en {i} iteraciones.")

        xprev, fprev = xcurr, fcurr
        xcurr, fcurr = x_next, fnext

    return _res(xcurr, max_iter, historial, False,
                f"No convergio en {max_iter} iteraciones.")


def _res(raiz, iteraciones, historial, convergio, justificacion):
    return {"raiz": raiz, "iteraciones": iteraciones, "historial": historial,
            "convergio": convergio, "justificacion": justificacion}


if __name__ == "__main__":
    f = lambda x: x**3 - x - 2
    res = secante(f, 1.0, 2.0, tol=1e-10)
    print(f"Raiz: {res['raiz']:.12f}  |  Iter: {res['iteraciones']}  |  f(raiz): {f(res['raiz']):.4e}")
