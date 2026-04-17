import math
import random
from typing import Callable, Any, Optional

Z_VALS = {
    "90%": 1.645,
    "95%": 1.960,
    "99%": 2.576
}

def estimar_pi(num_puntos: int, semilla: Optional[int] = None) -> dict[str, Any]:
    if semilla is not None:
        random.seed(semilla)
    
    puntos_dentro = 0
    puntos_grafica = []
    
    # Solo guardamos puntos de graficacion si son razonables (evitar desbordar streamlit)
    limite_puntos_guardados = 5000
    
    for i in range(num_puntos):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        dentro = (x**2 + y**2 <= 1)
        if dentro:
            puntos_dentro += 1
            
        if i < limite_puntos_guardados:
            puntos_grafica.append({
                "x": x, 
                "y": y, 
                "estado": "Dentro" if dentro else "Fuera"
            })
            
    pi_estimado = (puntos_dentro / num_puntos) * 4.0
    return {
        "pi_estimado": pi_estimado,
        "puntos_dentro": puntos_dentro,
        "num_puntos": num_puntos,
        "puntos_grafica": puntos_grafica
    }

def integracion_1d_mc(f: Callable[[float], float], a: float, b: float, num_puntos: int, confianza: str = "95%", semilla: Optional[int] = None) -> dict[str, Any]:
    if semilla is not None:
        random.seed(semilla)
        
    z = Z_VALS.get(confianza, 1.960)
    volumen = float(b - a)
    
    suma_f = 0.0
    cuadrados_f = 0.0
    
    puntos_grafica = []
    limite_puntos_guardados = 2000
    
    for i in range(num_puntos):
        xi = random.uniform(a, b)
        try:
            fi = float(f(xi))
        except Exception:
            fi = 0.0
        
        suma_f += fi
        cuadrados_f += fi**2
        
        if i < limite_puntos_guardados:
            puntos_grafica.append({"x": xi, "y": fi})
            
    # Promedios
    f_promedio = suma_f / num_puntos
    
    # Integral Estimada
    integral_estimada = volumen * f_promedio
    
    # Desviación estandar de la muestra
    if num_puntos > 1:
        varianza = (cuadrados_f - (suma_f**2) / num_puntos) / (num_puntos - 1)
        # Evitar flotantes negativos por redondeo numérico infinitesimal
        varianza = max(varianza, 0.0) 
        sigma = math.sqrt(varianza)
    else:
        sigma = 0.0
        
    error_estandar = sigma / math.sqrt(num_puntos)
    margen_error = volumen * z * error_estandar
    
    ic_inferior = integral_estimada - margen_error
    ic_superior = integral_estimada + margen_error

    return {
        "integral": integral_estimada,
        "margen_error": margen_error,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "f_promedio": f_promedio,
        "puntos_grafica": puntos_grafica
    }

def integracion_2d_mc(f_2d: Callable[[float, float], float], a: float, b: float, c: float, d: float, num_puntos: int, confianza: str = "95%", semilla: Optional[int] = None) -> dict[str, Any]:
    if semilla is not None:
        random.seed(semilla)
        
    z = Z_VALS.get(confianza, 1.960)
    area = float((b - a) * (d - c))
    
    suma_f = 0.0
    cuadrados_f = 0.0
    
    for i in range(num_puntos):
        xi = random.uniform(a, b)
        yi = random.uniform(c, d)
        
        try:
            fi = float(f_2d(xi, yi))
        except Exception:
            fi = 0.0
            
        suma_f += fi
        cuadrados_f += fi**2
            
    f_promedio = suma_f / num_puntos
    integral_estimada = area * f_promedio
    
    if num_puntos > 1:
        varianza = (cuadrados_f - (suma_f**2) / num_puntos) / (num_puntos - 1)
        varianza = max(varianza, 0.0)
        sigma = math.sqrt(varianza)
    else:
        sigma = 0.0
        
    error_estandar = sigma / math.sqrt(num_puntos)
    margen_error = area * z * error_estandar
    
    ic_inferior = integral_estimada - margen_error
    ic_superior = integral_estimada + margen_error

    return {
        "integral": integral_estimada,
        "margen_error": margen_error,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "f_promedio": f_promedio
    }

