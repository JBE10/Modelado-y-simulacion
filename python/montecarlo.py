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
    error_vs_pi = abs(pi_estimado - math.pi)
    # Desviación estándar del estimador de Bernoulli: p=pi/4, var=p(1-p)
    p_hat = puntos_dentro / num_puntos
    sigma_bernoulli = math.sqrt(p_hat * (1 - p_hat) / num_puntos) if num_puntos > 1 else 0.0
    ee_pi = 4.0 * sigma_bernoulli  # Error estándar de la estimación de pi
    return {
        "pi_estimado": pi_estimado,
        "puntos_dentro": puntos_dentro,
        "num_puntos": num_puntos,
        "error_vs_pi": error_vs_pi,
        "error_estandar": ee_pi,
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
    
    # Snapshots de convergencia progresiva (100 puntos equiespaciados)
    n_snapshots = 100
    paso_snapshot = max(1, num_puntos // n_snapshots)
    snapshots = []
    
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
        
        # Snapshot progresivo
        n_actual = i + 1
        if n_actual % paso_snapshot == 0 or n_actual == num_puntos:
            est_parcial = volumen * (suma_f / n_actual)
            if n_actual > 1:
                var_parcial = (cuadrados_f - (suma_f**2) / n_actual) / (n_actual - 1)
                var_parcial = max(var_parcial, 0.0)
                ee_parcial = math.sqrt(var_parcial) / math.sqrt(n_actual)
                margen_parcial = volumen * z * ee_parcial
            else:
                margen_parcial = 0.0
            snapshots.append({
                "n": n_actual,
                "integral": est_parcial,
                "ic_sup": est_parcial + margen_parcial,
                "ic_inf": est_parcial - margen_parcial,
            })
            
    # Promedios
    f_promedio = suma_f / num_puntos
    
    # Integral Estimada
    integral_estimada = volumen * f_promedio
    
    # Desviación estandar de la muestra
    if num_puntos > 1:
        varianza = (cuadrados_f - (suma_f**2) / num_puntos) / (num_puntos - 1)
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
        "sigma": sigma,
        "error_estandar": error_estandar,
        "snapshots": snapshots,
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
        "f_promedio": f_promedio,
        "sigma": sigma,
        "error_estandar": error_estandar
    }


def multi_run_1d(f: Callable[[float], float], a: float, b: float, num_puntos: int, k_corridas: int = 200, confianza: str = "95%") -> list[float]:
    """Ejecuta K corridas de MC sin seed fijo y devuelve la lista de integrales estimadas."""
    resultados = []
    for _ in range(k_corridas):
        r = integracion_1d_mc(f, a, b, num_puntos, confianza, semilla=None)
        resultados.append(r["integral"])
    return resultados

def simular_monty_hall(num_partidas: int, semilla: Optional[int] = None) -> dict[str, Any]:
    if semilla is not None:
        random.seed(semilla)
        
    wins_mantener = 0
    wins_cambiar = 0
    historial = []
    
    # limits the snapshot size to max 200 data points for UI graphing
    paso_snapshot = max(1, num_partidas // 200)
    
    for i in range(1, num_partidas + 1):
        puerta_premio = random.randint(0, 2)
        eleccion_inicial = random.randint(0, 2)
        
        # puertas disponibles para que abra el presentador
        opciones_presentador = [p for p in [0, 1, 2] if p != eleccion_inicial and p != puerta_premio]
        puerta_abierta = random.choice(opciones_presentador)
        
        # Estrategia mantener
        if eleccion_inicial == puerta_premio:
            wins_mantener += 1
            
        # Estrategia cambiar
        puerta_cambio = [p for p in [0, 1, 2] if p != eleccion_inicial and p != puerta_abierta][0]
        if puerta_cambio == puerta_premio:
            wins_cambiar += 1
            
        if i % paso_snapshot == 0 or i == num_partidas:
            historial.append({
                "partida": i,
                "win_rate_mantener": wins_mantener / i,
                "win_rate_cambiar": wins_cambiar / i
            })
            
    return {
        "num_partidas": num_partidas,
        "wins_mantener": wins_mantener,
        "wins_cambiar": wins_cambiar,
        "tasa_mantener": wins_mantener / num_partidas,
        "tasa_cambiar": wins_cambiar / num_partidas,
        "historial": historial
    }
