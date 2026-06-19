import time
import numpy as np

def run_test():
    # Simulate 600,000 steps of results
    n_steps = 600000
    res = {
        't': np.linspace(0, 300000, n_steps),
        'x': np.random.rand(n_steps) * 1e7,
        'z': np.random.rand(n_steps) * 1e7,
        'vx': np.random.rand(n_steps) * 7000,
        'vz': np.random.rand(n_steps) * 5000,
        'm': np.random.rand(n_steps) * 50000,
        'etapa': ["Boosters"] * 100000 + ["Core"] * 200000 + ["ICPS"] * 100000 + ["Carga Útil"] * 200000,
        'g_force': np.random.rand(n_steps) * 3,
        'q': np.random.rand(n_steps) * 80000
    }
    
    print("Simulated results generated. Profiling processing...")
    
    # 1. 3D coordinates calculation
    t0 = time.time()
    Re_m = 6371000.0
    phi = res["x"] / Re_m
    X_3d = (Re_m + res["z"]) * np.sin(phi)
    Y_3d = np.zeros_like(X_3d)
    Z_3d = (Re_m + res["z"]) * np.cos(phi)
    t1 = time.time()
    print(f"3D coords calc: {t1 - t0:.4f} s")
    
    # 2. Skimming loop (current implementation)
    t0 = time.time()
    n_total = len(X_3d)
    max_points = 2000
    skip_rate = max(1, n_total // max_points)
    trajectory_data = []
    for i in range(n_total):
        is_critical = res["t"][i] < 60
        if is_critical or (i % skip_rate == 0) or (i == n_total - 1):
            trajectory_data.append({
                "x": float(X_3d[i]), "y": float(Y_3d[i]), "z": float(Z_3d[i]),
                "f": res["etapa"][i], "v": float(np.sqrt(res["vx"][i]**2 + res["vz"][i]**2)),
                "t": float(res["t"][i]), "g": float(res["g_force"][i]), "q": float(res["q"][i])
            })
    t1 = time.time()
    print(f"Skimming loop: {t1 - t0:.4f} s")
    print(f"Trajectory points skimmed: {len(trajectory_data)}")
    
    # 3. 2D Map dataframe and altair preparation (current implementation)
    t0 = time.time()
    import pandas as pd
    phi_map = (res["x"] / 1000) / 6371.0
    r_map = 6371 + (res["z"] / 1000)
    traj_real_2d = pd.DataFrame({"x": r_map * np.sin(phi_map), "z": r_map * np.cos(phi_map), "Fase": res["etapa"]})
    t1 = time.time()
    print(f"2D map dataframe calc: {t1 - t0:.4f} s")
    
    # 4. Tabbed telemetry dataframe (current implementation)
    t0 = time.time()
    df_res = pd.DataFrame({
        "Tiempo (s)": res["t"], "Altitud (km)": res["z"] / 1000.0,
        "Velocidad (m/s)": np.sqrt(res["vx"]**2 + res["vz"]**2),
        "Fuerza G": res["g_force"], "Presión Dinámica (kPa)": res["q"] / 1000.0,
        "Masa (kg)": res["m"], "Etapa": res["etapa"]
    })
    t1 = time.time()
    print(f"Tabbed telemetry dataframe calc: {t1 - t0:.4f} s")

if __name__ == "__main__":
    run_test()
