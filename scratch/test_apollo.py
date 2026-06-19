import sys
sys.path.append('python')
import numpy as np
import time
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator, SimulationEngine

def run_test():
    # Setup Apollo 11 preset
    rocket = Rocket(payload_mass=45000)
    rocket.add_stage(RocketStage("S-IC (Etapa 1)", thrust=34000000, prop_mass=2100000, dry_mass=130000, burn_rate=13000))
    rocket.add_stage(RocketStage("S-II (Etapa 2)", thrust=5150000, prop_mass=450000, dry_mass=36000, burn_rate=1100))
    rocket.add_stage(RocketStage("S-IVB (Etapa 3 / TLI)", thrust=1000000, prop_mass=110000, dry_mass=10000, burn_rate=230))
    
    physics = EarthRocketPhysics()
    integrator = RK4Integrator()
    engine = SimulationEngine(rocket, physics, integrator)
    
    print("Starting simulation for Apollo 11 with t_ballistic=300000...")
    start_time = time.time()
    try:
        res = engine.run(h_step=0.5, t_ballistic=300000)
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.4f} seconds.")
        print(f"Number of steps: {len(res['t'])}")
        print(f"Final t: {res['t'][-1]} s")
        print(f"Min alt: {np.min(res['z'])} m")
        print(f"Max alt: {np.max(res['z'])} m")
        print(f"Final z: {res['z'][-1]} m")
    except Exception as e:
        print(f"Error during simulation: {e}")

if __name__ == "__main__":
    run_test()
