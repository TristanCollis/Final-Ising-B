# Final-Ising-B
 UCSD Winter 2024, Phys 142 Final Project - Ising 2D MCMC, Group B 

# Instructions
To run in either mode, 

``cd ./Final-Ising-B/source/``  
``python main simulate [args]``

## Running the Simulation  
The command-line arguments are as follows:  
  --label  
        str: The path at which to store or load the simulation data.  
  
  --lattice_size  
        int: The side length of the lattice (the total number of cells will be lattice_size^2)  
  
  --burn_in_steps  
        int: Number of burn-in steps.  
  
  --total_steps -steps  
        int: The total number of steps for which to run the simulation.  
  
  --samples SAMPLES
        int: The number of samples to take along the temperature and b-field axes.  
  
  --temperature_range, -tr  
        int, int: the lower and upper bounds of the temperature to sample.  

  --magnetic_field_range, -br  
        int, int: lower and upper bounds of the external b-field to sample.  
  
  --graph, -g  
        flag: if present, graphs the value  
  
  --force, -f  
        flag: If present, overwrites existing files at [label]
