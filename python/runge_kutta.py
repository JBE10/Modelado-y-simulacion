def euler_method(f, t0, y0, tf, h):
    """
    Método de Euler (Runge-Kutta de orden 1 / Taylor de orden 1).
    Resuelve y' = f(t, y) con valor inicial y(t0) = y0.
    """
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser mayor que 0.")
    if tf <= t0:
        raise ValueError("El tiempo final tf debe ser mayor que t0.")

    t_vals = [t0]
    y_vals = [y0]

    t = t0
    y = y0

    while t < tf - 1e-9:
        if t + h > tf:
            h = tf - t  # Ajuste del último paso para no exceder tf
        try:
            k1 = f(t, y)
            y = y + h * k1
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
        except Exception as e:
            raise RuntimeError(f"Error evaluando f(t, y) en t={t}, y={y}: {e}")

    return {
        "metodo": "Euler",
        "t_vals": t_vals,
        "y_vals": y_vals
    }


def rk2_method(f, t0, y0, tf, h, variante="Punto Medio"):
    """
    Método de Runge-Kutta de orden 2.
    Variantes soportadas: "Heun" (Trapecio), "Punto Medio", "Ralston".
    """
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser mayor que 0.")
    if tf <= t0:
        raise ValueError("El tiempo final tf debe ser mayor que t0.")

    if variante == "Heun":
        alpha = 1.0  # k1 en t, k2 en t+h
        c1, c2 = 0.5, 0.5
        p1, q11 = 1.0, 1.0
    elif variante == "Ralston":
        alpha = 2/3
        c1, c2 = 0.25, 0.75
        p1, q11 = 2/3, 2/3
    else:  # Por defecto Punto Medio (alpha = 0.5)
        alpha = 0.5
        c1, c2 = 0.0, 1.0
        p1, q11 = 0.5, 0.5

    t_vals = [t0]
    y_vals = [y0]

    t = t0
    y = y0

    while t < tf - 1e-9:
        if t + h > tf:
            h = tf - t
        try:
            k1 = f(t, y)
            k2 = f(t + p1 * h, y + q11 * h * k1)
            y = y + h * (c1 * k1 + c2 * k2)
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
        except Exception as e:
            raise RuntimeError(f"Error evaluando f(t, y) en t={t}, y={y}: {e}")

    return {
        "metodo": f"RK2 ({variante})",
        "t_vals": t_vals,
        "y_vals": y_vals
    }


def rk4_method(f, t0, y0, tf, h):
    """
    Método de Runge-Kutta clásico de orden 4 (RK4).
    """
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser mayor que 0.")
    if tf <= t0:
        raise ValueError("El tiempo final tf debe ser mayor que t0.")

    t_vals = [t0]
    y_vals = [y0]

    t = t0
    y = y0

    while t < tf - 1e-9:
        if t + h > tf:
            h = tf - t
        try:
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + h
            t_vals.append(t)
            y_vals.append(y)
        except Exception as e:
            raise RuntimeError(f"Error evaluando f(t, y) en t={t}, y={y}: {e}")

    return {
        "metodo": "RK4 Clásico",
        "t_vals": t_vals,
        "y_vals": y_vals
    }
