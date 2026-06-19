import sys
sys.path.append('python')
import numpy as np
import time
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator

def run_sim_standard(h_step=0.5, t_ballistic=300000):
    rocket = Rocket(payload_mass=26520)
    rocket.add_stage(RocketStage("Boosters + Core (Etapa 1)", thrust=39100000, prop_mass=1600000, dry_mass=293000, burn_rate=12000))
    rocket.add_stage(RocketStage("Core Stage (Etapa 2)", thrust=9000000, prop_mass=600000, dry_mass=98000, burn_rate=2000))
    rocket.add_stage(RocketStage("ICPS (Etapa 3 / TLI)", thrust=110100, prop_mass=26850, dry_mass=3800, burn_rate=24))
    
    physics = EarthRocketPhysics()
    integrator = RK4Integrator()
    
    # Run loop
    all_times = []
    all_states = []
    
    t = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Powered stages
    for idx, stage in enumerate(rocket.stages):
        current_prop = float(stage.prop_mass)
        burn_time = stage.prop_mass / stage.burn_rate
        t_end = t + burn_time
        while t < t_end:
            dt = min(h_step, t_end - t)
            m = rocket.get_current_mass(idx, current_prop)
            def f_wrap(t_val, s_val, _thrust=stage.thrust, _br=stage.burn_rate, _m=m):
                return physics.get_derivatives(t_val, s_val, _thrust, _br, _m)
            state = integrator.step(f_wrap, t, state, dt)
            current_prop -= stage.burn_rate * dt
            t += dt
            if state[1] < 0:
                state[1] = 0; state[3] = 0
            all_times.append(t)
            all_states.append(list(state))
            
    # Ballistic
    t_final = t + t_ballistic
    while t < t_final:
        dt = min(h_step, t_final - t)
        m = rocket.payload_mass
        def f_bal(t_val, s_val, _m=m):
            return physics.get_derivatives(t_val, s_val, 0.0, 0.0, _m)
        state = integrator.step(f_bal, t, state, dt)
        t += dt
        if state[1] < 0:
            state[1] = 0; state[3] = 0
            all_times.append(t)
            all_states.append(list(state))
            break
        all_times.append(t)
        all_states.append(list(state))
        
    return np.array(all_times), np.array(all_states)

def run_sim_adaptive(h_step=0.5, t_ballistic=300000, space_step=30.0):
    rocket = Rocket(payload_mass=26520)
    rocket.add_stage(RocketStage("Boosters + Core (Etapa 1)", thrust=39100000, prop_mass=1600000, dry_mass=293000, burn_rate=12000))
    rocket.add_stage(RocketStage("Core Stage (Etapa 2)", thrust=9000000, prop_mass=600000, dry_mass=98000, burn_rate=2000))
    rocket.add_stage(RocketStage("ICPS (Etapa 3 / TLI)", thrust=110100, prop_mass=26850, dry_mass=3800, burn_rate=24))
    
    physics = EarthRocketPhysics()
    integrator = RK4Integrator()
    
    # Run loop
    all_times = []
    all_states = []
    
    t = 0.0
    state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Powered stages
    for idx, stage in enumerate(rocket.stages):
        current_prop = float(stage.prop_mass)
        burn_time = stage.prop_mass / stage.burn_rate
        t_end = t + burn_time
        while t < t_end:
            dt = min(h_step, t_end - t)
            m = rocket.get_current_mass(idx, current_prop)
            def f_wrap(t_val, s_val, _thrust=stage.thrust, _br=stage.burn_rate, _m=m):
                return physics.get_derivatives(t_val, s_val, _thrust, _br, _m)
            state = integrator.step(f_wrap, t, state, dt)
            current_prop -= stage.burn_rate * dt
            t += dt
            if state[1] < 0:
                state[1] = 0; state[3] = 0
            all_times.append(t)
            all_states.append(list(state))
            
    # Ballistic with adaptive step in space
    t_final = t + t_ballistic
    while t < t_final:
        # If in space (z >= 120km), use space_step, else h_step
        curr_dt = space_step if state[1] >= 120000.0 else h_step
        dt = min(curr_dt, t_final - t)
        
        m = rocket.payload_mass
        def f_bal(t_val, s_val, _m=m):
            return physics.get_derivatives(t_val, s_val, 0.0, 0.0, _m)
        state = integrator.step(f_bal, t, state, dt)
        t += dt
        if state[1] < 0:
            state[1] = 0; state[3] = 0
            all_times.append(t)
            all_states.append(list(state))
            break
        all_times.append(t)
        all_states.append(list(state))
        
    return np.array(all_times), np.array(all_states)

def test():
    print("Running standard simulation...")
    t0 = time.time()
    t_std, s_std = run_sim_standard()
    t1 = time.time()
    print(f"Standard simulation completed in {t1 - t0:.4f} s. Steps: {len(t_std)}")
    print(f"Standard final altitude: {s_std[-1, 1]:.4f} m, final velocity: {np.sqrt(s_std[-1, 2]**2 + s_std[-1, 3]**2):.4f} m/s")
    
    print("\nRunning adaptive simulation...")
    t0 = time.time()
    t_adp, s_adp = run_sim_adaptive()
    t1 = time.time()
    print(f"Adaptive simulation completed in {t1 - t0:.4f} s. Steps: {len(t_adp)}")
    print(f"Adaptive final altitude: {s_adp[-1, 1]:.4f} m, final velocity: {np.sqrt(s_adp[-1, 2]**2 + s_adp[-1, 3]**2):.4f} m/s")
    
    # Calculate difference
    diff_alt = abs(s_std[-1, 1] - s_adp[-1, 1])
    diff_vel = abs(np.sqrt(s_std[-1, 2]**2 + s_std[-1, 3]**2) - np.sqrt(s_adp[-1, 2]**2 + s_adp[-1, 3]**2))
    print(f"\nDifference in final altitude: {diff_alt:.4f} m ({diff_alt / s_std[-1, 1] * 100:.6f}%)")
    print(f"Difference in final velocity: {diff_vel:.4f} m/s")

if __name__ == "__main__":
    test()
