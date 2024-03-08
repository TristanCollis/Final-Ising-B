import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rand
from numba import jit

 
def checker_grid(size):
    grid = np.empty((size, size))

    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                grid[i, j] = +1
            else:
                if(np.random.choice([0,0,0,0,1,1]) == 1): #make the grid more +1 spin sided
                    grid[i, j] = +1
                else:
                    grid[i,j]  = -1
    
    return grid


@jit(nopython=True)
def monte_carlo_step(spin_grid, temperature, b_field):
    i, j = np.random.randint(spin_grid.shape[0]), np.random.randint(spin_grid.shape[1])
    spin = spin_grid[i, j]
    neighbor_sum = (
          spin_grid[(i + 1) % size, j]
        + spin_grid[i, (j + 1) % size]
        + spin_grid[(i - 1) % size, j]
        + spin_grid[i, (j - 1) % size]
    )
    energy_diff = 2 * spin * neighbor_sum - b_field * spin
    if energy_diff <= 0 or np.random.rand() < np.exp(-energy_diff / temperature):
        spin_grid[i, j] *= -1


# Function to run the simulation
@jit(nopython=True)
def run_simulation(size, temperature, steps, spin_grid, b_field):
    energies  = []
    magnetism = []
    
    for i in range(steps):
        monte_carlo_step(spin_grid, temperature, b_field)
        
        magnetism.append(calculate_magnetism(spin_grid))
        energies.append(calculate_energy(spin_grid, b_field))
    return spin_grid, energies, magnetism

# Function to calculate net magnetism per generation
@jit(nopython=True)
def calculate_magnetism(spin_grid):  
    netMagnetism = np.sum(spin_grid)  
    return netMagnetism / spin_grid.size
    
# Function to calculate the energy of the spin configuration
@jit(nopython=True)
def calculate_energy(spin_grid, b_field):
    energy = 0
    grid_sum = np.sum(spin_grid)
    for i in range(size):
        for j in range(size):
            spin = spin_grid[i, j]
            neighbor_sum = (
                  spin_grid[(i + 1) % size, j]
                + spin_grid[i, (j + 1) % size]
                + spin_grid[(i - 1) % size, j]
                + spin_grid[i, (j - 1) % size]
            )
            energy += -spin * neighbor_sum - b_field * grid_sum
    return energy / 2

# Parameters
size = 100
temperature = np.linspace(1, 5, 15)
B_field = np.linspace(-1.5, 1.5, 15)
steps = 200000


mean_magnetization    = []
std_dev_magnetization = []
mean_energies         = []
std_dev_energies      = []

initial_grid = checker_grid(size)

#plot initial grid
plt.figure(figsize=(7, 7))
plt.imshow(initial_grid, cmap='binary', interpolation='nearest')
plt.title('Spin Initial Configuration (Black: +1, White: -1)')
plt.axis('off')
plt.show()



# Run simulation, 3D meshgrid
T, B = np.meshgrid(temperature, B_field)
M = np.zeros_like(T)  # Initialize magnetization array

for j, b_field in enumerate(B_field):
    print(j)
    for i, temp in enumerate(temperature):
        spin_grid = copy.deepcopy(initial_grid)

        
        spin_grid, energies, magnetism = run_simulation(size, temp, steps, spin_grid, b_field)

        #calculate the mean and std for M and E
        M_mean= np.mean(magnetism)
        mean_magnetization.append(np.mean(magnetism))
        M[j,i] =M_mean #each point on the meshgrid of external B's and temperatures has an associated mangetism now
        std_dev_magnetization.append(np.std(magnetism))
        mean_energies.append(np.mean(energies))
        std_dev_energies.append(np.std(energies))

        

# Create 3D plots
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, B, M, cmap='viridis')
ax.view_init(elev=30, azim=0)
ax.set_xlabel('Temperature')
ax.set_ylabel('B Field')
ax.set_zlabel('Magnetization')
ax.set_title('3D Meshgrid Display')
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, B, M, cmap='viridis')
ax.view_init(elev=30, azim=45)
ax.set_xlabel('Temperature')
ax.set_ylabel('B Field')
ax.set_zlabel('Magnetization')
ax.set_title('3D Meshgrid Display')
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, B, M, cmap='viridis')
ax.view_init(elev=30, azim=90)
ax.set_xlabel('Temperature')
ax.set_ylabel('B Field')
ax.set_zlabel('Magnetization')
ax.set_title('3D Meshgrid Display')
plt.show()

