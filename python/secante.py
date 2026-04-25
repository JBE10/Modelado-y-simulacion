import math
import sys
from typing import Callable, Any

from utils import _res

from utils import _res

def secante(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-7, max_iter: int = 100) -> dict[str, Any]:
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
        error = abs(xcurr - xprev)
        historial.append({"iter": i, "x_n": xprev, "f(x_n)": fprev, "x_{n+1}": xcurr, "error": error})
        
        if error <= tol:
            return _res(xcurr, i - 1, historial, True, f"Convergio prematuramente (cancelación evitada) en iter {i}.")

        denom = fcurr - fprev
        if abs(denom) <= sys.float_info.epsilon:
            return _res(xcurr, i - 1, historial, False, f"Denominador muy pequeño/casi nulo en iter {i}.")

        x_next = xcurr - fcurr * (xcurr - xprev) / denom
        if not math.isfinite(x_next):
            return _res(xcurr, i - 1, historial, False, f"x_(n+1) no es finito en iter {i}.")

        try:
            fnext = float(f(x_next))
        except Exception as e:
            return _res(xcurr, i - 1, historial, False, f"No se pudo evaluar f(x_(n+1)) en iter {i}. {e}")

        if not math.isfinite(fnext):
            return _res(x_next, i, historial, False, f"f(x_(n+1)) no es finito en iter {i}: {fnext}.")
        if abs(fnext) < tol:
            return _res(x_next, i, historial, True, f"Convergio en {i} iteraciones.")

        xprev, fprev = xcurr, fcurr
        xcurr, fcurr = x_next, fnext

    return _res(xcurr, max_iter, historial, False,
                f"No convergio tras {max_iter} iter. Error final: {abs(xcurr - xprev):.4e}")


if __name__ == "__main__":
    f = lambda x: x**3 - x - 2
    res = secante(f, 1, 2, tol=1e-10, max_iter=50)
    print(f"Raiz: {res['raiz']:.12f}  |  Iter: {res['iteraciones']}  |  f(raiz): {f(res['raiz']):.4e}")
