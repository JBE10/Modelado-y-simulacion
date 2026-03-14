def biseccion(
    f,
    a,
    b,
    tol=1e-7,
    max_iter=100,
    mostrar_tabla=False,
    devolver_historial=False,
):
    """
    Encuentra una raiz de f(x)=0 en [a, b] usando el metodo de biseccion.

    Parametros:
    - f: funcion continua
    - a, b: extremos del intervalo (deben cumplir f(a)*f(b) < 0)
    - tol: tolerancia para el error
    - max_iter: maximo numero de iteraciones
    - mostrar_tabla: si es True, imprime una tabla con cada iteracion
    - devolver_historial: si es True, retorna tambien una lista con el detalle de iteraciones

    Retorna:
    - (raiz_aproximada, numero_iteraciones)
    - (raiz_aproximada, numero_iteraciones, historial) si devolver_historial=True
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("El intervalo [a, b] no encierra una raiz (f(a) y f(b) mismo signo).")

    if mostrar_tabla:
        print(f"{'Iter':>4} | {'a':>12} | {'b':>12} | {'c':>12} | {'f(c)':>12} | {'error':>12}")
        print("-" * 79)

    historial = []

    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0

        historial.append(
            {
                "iter": i,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error,
            }
        )

        if mostrar_tabla:
            print(f"{i:4d} | {a:12.8f} | {b:12.8f} | {c:12.8f} | {fc:12.4e} | {error:12.4e}")

        # Criterios de paro: residuo pequeno o intervalo suficientemente pequeno.
        if abs(fc) < tol or error < tol:
            if devolver_historial:
                return c, i, historial
            return c, i

        # Elegir subintervalo que conserva el cambio de signo.
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raiz = (a + b) / 2.0
    if devolver_historial:
        return raiz, max_iter, historial
    return raiz, max_iter


if __name__ == "__main__":
    def f(x):
        return x**3 - x - 2  # Raiz real aproximada: 1.52138

    raiz, iters = biseccion(f, 1, 2, tol=1e-10, max_iter=200, mostrar_tabla=True)
    print("Raiz aproximada:", raiz)
    print("Iteraciones:", iters)
    print("f(raiz):", f(raiz))
