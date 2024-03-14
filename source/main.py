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
    label: str = args.label

    match args.subparsers:
        case "simulate":
            do_simulation: bool = True
            do_analyze: bool = False

            do_graph: bool = args.graph
            force: bool = args.force

            lattice_size: int = args.lattice_size
            burn_in_steps: int = args.burn_in_steps
            total_steps: int = args.total_steps
            samples: int = args.samples
            temperature_range: list[float] = args.temperature_range
            b_field_range: list[float] = args.magnetic_field_range

        case "analyze":
            do_analyze: bool = True
            do_simulation: bool = False

            do_graph: bool = args.graph

            total_steps: int = args.total_steps
            temperature_index: int = args.temperature_index
            magnetic_field_index: int = args.magnetic_field_index
            percent: float = args.percent
            error: float = args.error

    if label == "":
        print("Please provide a label for the simulation")
        sys.exit()

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

    if do_simulation:
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
            np.savetxt(sim_path / "temperatures.csv", temperatures, delimiter=",")
            np.savetxt(sim_path / "bfields.csv", b_fields, delimiter=",")
            np.savez(sim_path / "history", temperature=temperatures, bfield=b_fields, magnetization_history=magnetization_history)

        if do_graph:
            logger.info("Starting graphing simulation")
            magnetization_with_burn = np.mean(magnetization_history[:, :, burn_in_steps:], axis=2)

            T, B = np.meshgrid(temperatures, b_fields)

            plot_3D(T, B, magnetization_with_burn, 0, sim_path)
            plot_3D(T, B, magnetization_with_burn, 45, sim_path)
            plot_3D(T, B, magnetization_with_burn, 90, sim_path)
            logger.info("Graphing simulation complete")

    if do_analyze:
        logger.info("Starting analysis")

        saved_arrays = np.load(sim_path / "history.npz")
        magnetization_history = saved_arrays['magnetization_history']
        temperatures = saved_arrays['temperature']
        b_fields = saved_arrays['bfield']
        
        analyze_timer = Timer("Analysis")

        magnetization = magnetization_history[temperature_index, magnetic_field_index]
        burnout_limit = burnout(total_steps, percent, error, magnetization)

        analyze_timer.stop()

        if(np.isnan(burnout_limit)):
            logger.info("Burnout limit not found")
        else:
            logger.info(f"Burnout limit: {burnout_limit}")

        if do_graph:
            plot_burnout(magnetization, burnout_limit, sim_path, temperatures[temperature_index], b_fields[magnetic_field_index])
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        type=str,
        default=""
    )

    subparsers = parser.add_subparsers(dest="subparsers")
    
    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument(
        "--lattice_size", 
        type=int, 
        default="10"
    )
    simulate_parser.add_argument(
        "--burn_in_steps", 
        type=int, 
        default="1000"
    )
    simulate_parser.add_argument(
        "--total_steps",
        "-steps", 
        type=int, 
        default="5000"
    )
    simulate_parser.add_argument(
        "--samples", 
        type=int, 
        default="9"
    )
    simulate_parser.add_argument(
        "--temperature_range",
        "-tr",
        type=float,
        nargs=2,
        default=[1, 5]
    )
    simulate_parser.add_argument(
        "--magnetic_field_range",
        "-br",
        type=float,
        nargs=2,
        default=[-5, 5]
    )
    simulate_parser.add_argument(
        "--graph", 
        "-g",
        action='store_true'
    )
    simulate_parser.add_argument(
        "--force",
        "-f",
        action='store_true'
    )

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--total_steps",
        "-steps", 
        type=int, 
        default="10"
    )
    analyze_parser.add_argument(
        "--temperature_index",
        "-ti",
        type=int,
        required=True
    )
    analyze_parser.add_argument(
        "--magnetic_field_index",
        "-bi",
        type=int,
        required=True
    )
    analyze_parser.add_argument(
        "--percent",
        "-p",
        type=float,
        default="0.5"
    )
    analyze_parser.add_argument(
        "--error",
        "-e",
        type=float,
        default="0.05"
    )
    analyze_parser.add_argument(
        "--graph", 
        "-g",
        action='store_true'
    )

    args = parser.parse_args()
    main(args)
