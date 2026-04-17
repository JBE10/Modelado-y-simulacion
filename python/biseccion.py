import math
from typing import Callable, Any

from utils import _res

def biseccion(f: Callable[[float], float], a: float, b: float, tol: float = 1e-7, max_iter: int = 100) -> dict[str, Any]:
    a = float(a)
    b = float(b)
    fa = float(f(a))
    fb = float(f(b))

    if not math.isfinite(fa) or not math.isfinite(fb):
        raise ValueError("f(a) o f(b) no es finito.")
    if abs(fa) < tol:
        return _res(a, 0, [], True, f"Convergio en 0 iter: |f(a)| = {abs(fa):.4e} < tol.")
    if abs(fb) < tol:
        return _res(b, 0, [], True, f"Convergio en 0 iter: |f(b)| = {abs(fb):.4e} < tol.")

    if fa * fb > 0:
        raise ValueError("f(a) y f(b) tienen el mismo signo: no hay raiz garantizada en [a,b].")

    historial = []
    for i in range(1, max_iter + 1):
        c = a + (b - a) / 2.0
        fc = float(f(c))
        error = abs(b - a) / 2.0
        historial.append({"iter": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": error})

        if not math.isfinite(fc):
            return _res(c, i, historial, False, f"f(c) no es finito en iter {i}: {fc}.")
        if abs(fc) < tol or abs(error) < tol:
            return _res(c, i, historial, True, f"Convergio en {i} iteraciones.")

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raiz = a + (b - a) / 2.0
    n_est = math.ceil(math.log2((b - a) / tol)) if tol > 0 else max_iter
    return _res(raiz, max_iter, historial, False, f"No convergio en {max_iter} iter. Se estiman ~{n_est} para tol={tol:.0e}.")


if __name__ == "__main__":
    f = lambda x: x**3 - x - 2
    res = biseccion(f, 1, 2, tol=1e-10, max_iter=200)
    print(f"Raiz: {res['raiz']:.12f}  |  Iter: {res['iteraciones']}  |  f(raiz): {f(res['raiz']):.4e}")
