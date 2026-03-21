import math


def newton_raphson(f, df, x0, tol=1e-7, max_iter=100):
    historial = []
    xn = float(x0)

    try:
        fx = float(f(xn))
    except Exception as e:
        return _res(xn, 0, historial, False, f"No se pudo evaluar f(x0). {e}")

    if not math.isfinite(fx):
        return _res(xn, 0, historial, False, f"f(x0) no es finito: {fx}.")
    if abs(fx) < tol:
        return _res(xn, 0, historial, True, f"Convergio en 0 iter: |f(x0)| = {abs(fx):.4e} < tol.")

    for i in range(1, max_iter + 1):
        try:
            dfx = float(df(xn))
        except Exception as e:
            return _res(xn, i - 1, historial, False, f"Error evaluando f'(x_n) en iter {i}. {e}")

        if not math.isfinite(dfx):
            return _res(xn, i - 1, historial, False, f"f'(x_n) no es finito en iter {i}: {dfx}.")
        if abs(dfx) < 1e-14:
            return _res(xn, i - 1, historial, False, f"Derivada casi nula en iter {i}.")

        x_next = xn - fx / dfx
        if not math.isfinite(x_next):
            return _res(xn, i - 1, historial, False, f"x_(n+1) no es finito en iter {i}.")

        error = abs(x_next - xn)
        try:
            fx_next = float(f(x_next))
        except Exception as e:
            return _res(xn, i - 1, historial, False, f"Error evaluando f(x_(n+1)) en iter {i}. {e}")

        historial.append({
            "iter": i,
            "x_n": xn,
            "f(x_n)": fx,
            "f'(x_n)": dfx,
            "x_n+1": x_next,
            "error": error,
        })

        if not math.isfinite(fx_next):
            return _res(x_next, i, historial, False, f"f(x_(n+1)) no es finito en iter {i}: {fx_next}.")
        if abs(fx_next) < tol or error < tol:
            return _res(x_next, i, historial, True, f"Convergio en {i} iteraciones.")

        xn, fx = x_next, fx_next

    return _res(xn, max_iter, historial, False,
                f"No convergio en {max_iter} iteraciones.")


def _res(raiz, iteraciones, historial, convergio, justificacion):
    return {"raiz": raiz, "iteraciones": iteraciones, "historial": historial,
            "convergio": convergio, "justificacion": justificacion}


if __name__ == "__main__":
    f = lambda x: x**3 - x - 2
    df = lambda x: 3 * x**2 - 1
    res = newton_raphson(f, df, 1.5, tol=1e-10)
    print(f"Raiz: {res['raiz']:.12f}  |  Iter: {res['iteraciones']}  |  f(raiz): {f(res['raiz']):.4e}")
