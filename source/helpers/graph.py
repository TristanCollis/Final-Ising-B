import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("main")
plt.ioff()

def plot_3D(T, B, M, azim, path, elev: int = 30):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, B, M, cmap='viridis') #type: ignore
    ax.view_init(elev=elev, azim=azim) #type: ignore
    ax.set_xlabel('Temperature')
    ax.set_ylabel('B Field')
    ax.set_zlabel('Magnetization') #type: ignore
    ax.set_title('3D Meshgrid Display')
    logger.info(f"Saving graph_{azim}.png at {path / f'graph_{azim}.png'}")
    plt.savefig(path / f"graph_{azim}.png")


def plot_burnout(magnetism, burnout_limit, path, temperature, b_field):
    plt.plot(magnetism)
    plt.xlabel("steps")
    plt.ylabel("Net Magnetization")
    plt.axvline(
        x=burnout_limit, color="r", linestyle="--"
    )  # Adjust color and linestyle as needed
    plt.grid(True)
    plt.title(
        f"Burnout Limit: {burnout_limit} \n Temperature: {temperature} \n B Field: {b_field}"
    )
    logger.info(f"Saving burnout.png at {path / 'burnout.png'}")
    plt.savefig(path / "burnout.png")
