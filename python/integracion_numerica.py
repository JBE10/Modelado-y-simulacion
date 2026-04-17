import numpy as np


def integral_referencia(f, a, b, limit=500, n_trap_fallback=50_001):
    """
    Valor de referencia de ∫_a^b f(x) dx para comparar con Newton-Cotes.

    Si scipy está instalado: cuadratura adaptativa (quad) y su estimación de error.
    Si no: trapecio compuesto con muchos subintervalos (sin cota de error fiable).
    """
    try:
        from scipy.integrate import quad

        val, err = quad(f, a, b, limit=limit)
        return float(val), float(err)
    except ImportError:
        x = np.linspace(a, b, n_trap_fallback)
        y = np.array([float(f(xi)) for xi in x], dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError("f(x) no finita en parte del intervalo; no hay referencia.")
        h = (b - a) / (n_trap_fallback - 1)
        val = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        return float(val), float("nan")

def integracion_rectangulo(f, a, b, n):
    """
    Regla compuesta del rectángulo (Punto Medio).
    """
    if n < 1:
        raise ValueError("n debe ser al menos 1")
    
    h = (b - a) / n
    # Puntos medios
    x_m = np.linspace(a + h/2, b - h/2, n)
    try:
        y_m = np.array([f(x) for x in x_m], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Error evaluando f(x): {e}")

    valor = h * np.sum(y_m)
    
    # Para visualización: puntos donde evaluamos
    x_vals = x_m.tolist()
    y_vals = y_m.tolist()
    
    return {
        "metodo": "Rectángulo M.",
        "h": h,
        "valor": valor,
        "x_vals": x_vals,
        "y_vals": y_vals
    }

def integracion_trapecio(f, a, b, n):
    """
    Regla compuesta del trapecio.
    """
    if n < 1:
        raise ValueError("n debe ser al menos 1")
        
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Error evaluando f(x): {e}")

    valor = (h / 2.0) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1])
    
    return {
        "metodo": "Trapecio",
        "h": h,
        "valor": valor,
        "x_vals": x.tolist(),
        "y_vals": y.tolist()
    }

def integracion_simpson_13(f, a, b, n):
    """
    Regla compuesta de Simpson 1/3.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("Para Simpson 1/3, el número de subintervalos 'n' debe ser un entero PAR.")
        
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Error evaluando f(x): {e}")

    # y[1:-1:2] -> impares: 1, 3, 5...
    # y[2:-2:2] -> pares: 2, 4, 6...
    suma_impares = np.sum(y[1:-1:2])
    suma_pares = np.sum(y[2:-2:2])
    
    valor = (h / 3.0) * (y[0] + 4.0 * suma_impares + 2.0 * suma_pares + y[-1])
    
    return {
        "metodo": "Simpson 1/3",
        "h": h,
        "valor": valor,
        "x_vals": x.tolist(),
        "y_vals": y.tolist()
    }

def integracion_simpson_38(f, a, b, n):
    """
    Regla compuesta de Simpson 3/8.
    """
    if n < 3 or n % 3 != 0:
        raise ValueError("Para Simpson 3/8, el número de subintervalos 'n' debe ser MÚLTIPLO de 3.")
        
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x], dtype=float)
    except Exception as e:
        raise RuntimeError(f"Error evaluando f(x): {e}")

    # Necesitamos cuidado al separar los índices:
    # La suma es de y[i], para i no múltiplo de 3 (se multiplica por 3)
    # y para i múltiplo de 3 (se multiplica por 2)
    # y[0] y y[-1] corren solos.
    
    # Índices del 1 al n-1
    indices = np.arange(1, n)
    # Máscara para múltiplos de 3:
    mask_mult3 = (indices % 3 == 0)
    
    suma_resto = np.sum(y[indices[~mask_mult3]])
    suma_mult3 = np.sum(y[indices[mask_mult3]])
    
    valor = (3.0 * h / 8.0) * (y[0] + 3.0 * suma_resto + 2.0 * suma_mult3 + y[-1])
    
    return {
        "metodo": "Simpson 3/8",
        "h": h,
        "valor": valor,
        "x_vals": x.tolist(),
        "y_vals": y.tolist()
    }
