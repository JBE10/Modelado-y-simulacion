import sys
sys.path.append('python')
import numpy as np
import time
from rocket_engine_solid import Rocket, RocketStage, EarthRocketPhysics, RK4Integrator, SimulationEngine

def run_test():
    rocket = Rocket(payload_mass=26520)
    rocket.add_stage(RocketStage("Boosters + Core (Etapa 1)", thrust=39100000, prop_mass=1600000, dry_mass=293000, burn_rate=12000))
    rocket.add_stage(RocketStage("Core Stage (Etapa 2)", thrust=9000000, prop_mass=600000, dry_mass=98000, burn_rate=2000))
    rocket.add_stage(RocketStage("ICPS (Etapa 3 / TLI)", thrust=110100, prop_mass=26850, dry_mass=3800, burn_rate=24))
    
    physics = EarthRocketPhysics()
    integrator = RK4Integrator()
    engine = SimulationEngine(rocket, physics, integrator)
    
    print("Starting simulation with t_ballistic=300000...")
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
