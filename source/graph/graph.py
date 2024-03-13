import matplotlib.pyplot as plt
from pathlib import Path

def plot_3D(T, B, M, azim, path):
    print(len(T))
    print(len(B))
    print(M.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, B, M, cmap='viridis')
    ax.view_init(elev=30, azim=azim)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('B Field')
    ax.set_zlabel('Magnetization')
    ax.set_title('3D Meshgrid Display')
    plt.savefig(path / f"graph_{azim}.png")