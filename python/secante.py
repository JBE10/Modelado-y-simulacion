import math


def secante(
    f,
    x0,
    x1,
    tol=1e-7,
    max_iter=100,
    mostrar_tabla=False,
    devolver_historial=False,
):
    """
    Encuentra una raiz de f(x)=0 usando el metodo de la Secante.

    Usa dos aproximaciones iniciales x0, x1 y la recta secante
    para generar x_{n+1} sin necesidad de derivada explicita.
    """
    if tol <= 0:
        raise ValueError("La tolerancia debe ser mayor a 0.")
    if max_iter < 1:
        raise ValueError("max_iter debe ser >= 1.")

    historial = []
    xprev = float(x0)
    xcurr = float(x1)

    try:
        fprev = float(f(xprev))
    except Exception as e:
        raise ValueError(f"No se pudo evaluar f(x0): {e}") from e
    if not math.isfinite(fprev):
        raise ValueError("f(x0) no es finito (NaN/Inf).")

    if mostrar_tabla:
        print(f"{'Iter':>4} | {'x_{n-1}':>14} | {'x_n':>14} | {'f(x_n)':>12} | {'x_{n+1}':>14} | {'error':>12}")
        print("-" * 86)

    for i in range(1, max_iter + 1):
        try:
            fcurr = float(f(xcurr))
        except Exception as e:
            return _resultado(
                xcurr, i - 1, historial, False,
                f"Se detuvo en iter {i}: no se pudo evaluar f(x_n). Error: {e}",
                devolver_historial,
            )

        if not math.isfinite(fcurr):
            return _resultado(
                xcurr, i - 1, historial, False,
                f"Se detuvo en iter {i}: f(x_n) dio NaN/Inf.",
                devolver_historial,
            )

        denom = fcurr - fprev
        if abs(denom) < 1e-14:
            return _resultado(
                xcurr, i, historial, False,
                f"Se detuvo en iter {i}: f(x_n)-f(x_{{n-1}}) casi 0 (denominador degenerado).",
                devolver_historial,
            )

        x_next = xcurr - fcurr * (xcurr - xprev) / denom
        error = abs(x_next - xcurr)

        if not math.isfinite(x_next):
            return _resultado(
                xcurr, i, historial, False,
                f"Se detuvo en iter {i}: x_{{n+1}} dio NaN/Inf.",
                devolver_historial,
            )

        fila = {
            "iter": i,
            "x_{n-1}": xprev,
            "x_n": xcurr,
            "f(x_n)": fcurr,
            "x_{n+1}": x_next,
            "error": error,
        }
        historial.append(fila)

        if mostrar_tabla:
            print(f"{i:4d} | {xprev:14.8f} | {xcurr:14.8f} | {fcurr:12.4e} | {x_next:14.8f} | {error:12.4e}")

        if error < tol or abs(fcurr) < tol:
            return _resultado(
                x_next, i, historial, True,
                f"Convergio en iter {i}: se cumplio |x_{{n+1}} - x_n| < tol o |f(x_n)| < tol.",
                devolver_historial,
            )

        xprev = xcurr
        fprev = fcurr
        xcurr = x_next

    return _resultado(
        xcurr, max_iter, historial, False,
        f"No convergio: se alcanzo max_iter={max_iter}. Revisa x0 y x1.",
        devolver_historial,
    )


def _resultado(raiz, iteraciones, historial, convergio, justificacion, devolver_historial):
    out = {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "historial": historial,
        "convergio": convergio,
        "justificacion": justificacion,
    }
    if devolver_historial:
        return out
    return {k: v for k, v in out.items() if k != "historial"}


if __name__ == "__main__":
    def f(x):
        return x**3 - x - 2

    res = secante(f, 1.0, 2.0, tol=1e-10, max_iter=200, mostrar_tabla=True, devolver_historial=True)
    print(f"\nRaiz aproximada: {res['raiz']:.12f}")
    print(f"Iteraciones: {res['iteraciones']}")
    print(f"|f(raiz)|: {abs(f(res['raiz'])):.4e}")
    print(f"Convergio: {res['convergio']}")
