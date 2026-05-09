import numpy as np

def rocket_derivatives(t, Y, T, m_dot, Cd, A):
    """
    Computa las derivadas (pendientes) para el sistema de EDOs del cohete.
    Y = [x, z, vx, vz, m]
    """
    x, z, vx, vz, m = Y
    
    # Constantes físicas
    g0 = 9.81
    Re = 6371000 # Radio de la Tierra en metros
    rho0 = 1.225 # Densidad atmosférica a nivel del mar
    H = 8000 # Escala de altura atmosférica
    
    # Gravedad (disminuye con la altura)
    g = g0 * (Re / (Re + z))**2
    
    # Densidad atmosférica
    rho = rho0 * np.exp(-z / H) if z < 100000 else 0.0
    
    # Velocidad escalar
    v = np.sqrt(vx**2 + vz**2)
    
    # Fuerza de Arrastre Aerodinámico (Drag)
    D = 0.5 * rho * v**2 * Cd * A
    
    # Ángulo de cabeceo (Pitch) - Gravity Turn simplificado
    # Sube recto hasta t=10, luego gira progresivamente hacia horizontal
    if t < 10:
        theta = np.pi / 2
    else:
        theta = np.pi / 2 * (1 - (t - 10) / 140)
        theta = max(0.0, theta) # Limitar a 0 grados (horizontal)
        
    # EDOs Cinemáticas
    dxdt = vx
    dzdt = vz
    
    # EDOs Dinámicas (Aceleración)
    if v < 1e-6:
        dvxdt = (T * np.cos(theta)) / m
        dvzdt = (T * np.sin(theta)) / m - g
    else:
        dvxdt = (T * np.cos(theta) - D * (vx / v)) / m
        dvzdt = (T * np.sin(theta) - D * (vz / v)) / m - g
        
    # EDO de Masa (Consumo de combustible)
    dmdt = -m_dot
    
    return np.array([dxdt, dzdt, dvxdt, dvzdt, dmdt])

def rk4_step(f, t, y, h, *args):
    """Paso numérico usando el método de Runge-Kutta de orden 4."""
    k1 = f(t, y, *args)
    k2 = f(t + h/2, y + h/2 * k1, *args)
    k3 = f(t + h/2, y + h/2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_stage(t0, Y0, t_end, h, T, m_dot, Cd, A):
    """Simula una etapa del cohete hasta que se acaba el tiempo de quemado."""
    times = [t0]
    states = [Y0]
    
    t = t0
    y = np.array(Y0)
    
    while t < t_end:
        dt = min(h, t_end - t)
        y = rk4_step(rocket_derivatives, t, y, dt, T, m_dot, Cd, A)
        t += dt
        
        # Condición de suelo (si choca o aún no despega)
        if y[1] < 0:
            y[1] = 0
            y[3] = 0
            if y[2] < 0: y[2] = 0 # Fricción bruta con el suelo
            
        times.append(t)
        states.append(y)
        
    return times, states

def simular_lanzamiento(params):
    """
    Orquesta el PVI completo con separaciones de etapas (discontinuidades de masa).
    """
    # ── Parámetros de la Etapa 1 ──
    T1 = params.get('T1', 7600000) # Empuje (N)
    m1_prop = params.get('m1_prop', 395700) # Masa propelente (kg)
    m1_dry = params.get('m1_dry', 25600) # Masa seca
    m_dot1 = params.get('m_dot1', 2500) # Tasa de consumo
    t_burn1 = m1_prop / m_dot1
    
    # ── Parámetros de la Etapa 2 ──
    T2 = params.get('T2', 934000)
    m2_prop = params.get('m2_prop', 92670)
    m2_dry = params.get('m2_dry', 3900)
    m_dot2 = params.get('m_dot2', 260)
    t_burn2 = m2_prop / m_dot2
    
    # Payload
    m_payload = params.get('m_payload', 5000)
    
    Cd = 0.4
    A = 10.5
    h_step = params.get('h_step', 0.5)
    
    # Estado inicial: x=0, z=0, vx=0, vz=0, masa_total
    t0 = 0.0
    m0 = m1_prop + m1_dry + m2_prop + m2_dry + m_payload
    Y0 = [0.0, 0.0, 0.0, 0.0, m0]
    
    all_times = []
    all_states = []
    etapas_flags = []
    
    # ── 1. Simular Etapa 1 ──
    t1_end = t0 + t_burn1
    t1_arr, s1_arr = simulate_stage(t0, Y0, t1_end, h_step, T1, m_dot1, Cd, A)
    all_times.extend(t1_arr)
    all_states.extend(s1_arr)
    etapas_flags.extend(['Etapa 1 (Ignición)'] * len(t1_arr))
    
    # ── Separación de Etapa 1 ──
    t_sep = all_times[-1]
    Y_sep = np.array(all_states[-1])
    Y_sep[4] -= m1_dry # Se descarta la masa seca de la etapa 1
    
    # ── 2. Simular Etapa 2 ──
    t2_end = t_sep + t_burn2
    t2_arr, s2_arr = simulate_stage(t_sep, Y_sep, t2_end, h_step, T2, m_dot2, Cd, A)
    all_times.extend(t2_arr[1:])
    all_states.extend(s2_arr[1:])
    etapas_flags.extend(['Etapa 2 (Vuelo Orbital)'] * (len(t2_arr)-1))
    
    # ── Separación de Etapa 2 ──
    t_bal = all_times[-1]
    Y_bal = np.array(all_states[-1])
    Y_bal[4] -= m2_dry # Queda solo la carga útil (payload)
    
    # ── 3. Vuelo Balístico (Payload sin motores) ──
    t_end = t_bal + params.get('t_bal', 300) # simular 5 mins extra
    t3_arr, s3_arr = simulate_stage(t_bal, Y_bal, t_end, h_step, 0.0, 0.0, Cd, A/2)
    all_times.extend(t3_arr[1:])
    all_states.extend(s3_arr[1:])
    etapas_flags.extend(['Carga Útil (Vuelo Balístico)'] * (len(t3_arr)-1))
    
    res = {
        't': np.array(all_times),
        'x': np.array([s[0] for s in all_states]),
        'z': np.array([s[1] for s in all_states]),
        'vx': np.array([s[2] for s in all_states]),
        'vz': np.array([s[3] for s in all_states]),
        'm': np.array([s[4] for s in all_states]),
        'etapa': etapas_flags
    }
    
    return res
