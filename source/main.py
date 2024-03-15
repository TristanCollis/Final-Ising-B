import argparse
import sys
import logging
import os
from pathlib import Path

import numpy as np
from simulate import simulate, burnout
from helpers.graph import plot_3D, plot_burnout
from helpers.timer import Timer

def main(args):
    mode: str = args.mode

    label: str = args.label
    lattice_size: int = args.lattice_size
    burn_in_steps: int = args.burn_in_steps
    total_steps: int = args.total_steps
    samples: int = args.samples
    temperature_range: list[float] = args.temperature_range
    b_field_range: list[float] = args.magnetic_field_range
    magnetization_index: list[int] = args.magnetization_index
    percent: float = args.percent
    error: float = args.error

    force: bool = args.force

    path = Path(".")
    if not (path / "data").exists():
        (path / "data").mkdir()
    if not (path / "data"/ label).exists():
        (path / "data" / label).mkdir()
    sim_path = path / "data" / label

    log_path = sim_path / f"{label}.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    logger.info(f"Saving data to {sim_path}")

    match mode:
        case "simulate":
            if (sim_path / "history.npz").exists() and not force:
                logger.info("Existing simulation found and force flag not set, skipping simulation")
                saved_arrays = np.load(sim_path / "history.npz")
                temperatures = saved_arrays['temperature']
                b_fields = saved_arrays['bfield']
                magnetization_history = saved_arrays['magnetization_history']
            else:
                logger.info("Starting simulation")
                simulation_timer = Timer("Simulation")

                temperatures = np.linspace(temperature_range[0], temperature_range[1], num=samples)
                b_fields = np.linspace(b_field_range[0], b_field_range[1], num=samples)

                magnetization_history = simulate(
                    lattice_size, total_steps, temperatures, b_fields
                )

                simulation_timer.stop()
                logger.info("Simulation complete")

                np.savetxt(sim_path / "temperatures.csv", temperatures, delimiter=",")
                np.savetxt(sim_path / "bfields.csv", b_fields, delimiter=",")
                np.savez(sim_path / "history", temperature=temperatures, bfield=b_fields, magnetization_history=magnetization_history)

            magnetization_with_burn = np.mean(magnetization_history[:, :, burn_in_steps:], axis=2)
            T, B = np.meshgrid(temperatures, b_fields)
            plot_3D(T, B, magnetization_with_burn, 0, sim_path, elev=90)
            plot_3D(T, B, magnetization_with_burn, 45, sim_path)
            plot_3D(T, B, magnetization_with_burn, 90, sim_path)
            plot_3D(T, B, magnetization_with_burn, 315, sim_path)
        
        case "burnout":
            if not (sim_path / "history.npz").exists():
                logger.info("No simulation found, skipping burnout analysis")
                sys.exit()
            if magnetization_index is None:
                logger.info("No magnetization index provided, skipping burnout analysis")
                sys.exit()
            
            logger.info("Starting burnout analysis")

            saved_arrays = np.load(sim_path / "history.npz")
            magnetization_history = saved_arrays['magnetization_history']
            temperatures = saved_arrays['temperature']
            b_fields = saved_arrays['bfield']
            magnetization = magnetization_history[magnetization_index]
            burnout_limit = burnout(total_steps, percent, error, magnetization)
            if(np.isnan(burnout_limit)):
                logger.info("Burnout limit not found")
            else:
                logger.info(f"Burnout limit: {burnout_limit}")
            
            plot_burnout(magnetization, burnout_limit, sim_path, temperatures[magnetization_index[0]], b_fields[magnetization_index[1]])                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here")

    parser.add_argument("mode", choices=["simulate", "burnout"], help="Mode of operation")
    parser.add_argument("--label", "-l", type=str, default="", help="Label string", required=True)
    parser.add_argument("--lattice_size", type=int, default=10, help="Size of the lattice")
    parser.add_argument("--burn_in_steps", type=int, default=1000, help="Number of burn-in steps")
    parser.add_argument("--total_steps", type=int, default=5000, help="Total number of steps")
    parser.add_argument("--samples", type=int, default=9, help="Number of samples")
    parser.add_argument("--temperature_range", nargs=2, type=int, default=[1, 5], help="Temperature range")
    parser.add_argument("--magnetic_field_range", nargs=2, type=int, default=[-5, 5], help="Magnetic field range")
    parser.add_argument("--magnetization_index", type=int, default=2, help="Magnetization index")
    parser.add_argument("--percent", type=float, default=0.5, help="Percent")
    parser.add_argument("--error", type=float, default=0.05, help="Error")
    parser.add_argument("--force", "-f", action="store_true", help="Force flag")

    main(parser.parse_args())