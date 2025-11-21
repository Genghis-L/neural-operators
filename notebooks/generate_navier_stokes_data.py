#!/usr/bin/env python3
"""
Generate Navier-Stokes data file: NavierStokes_V1e-5_N1200_T20.mat

This script generates data for the 2D Navier-Stokes equation with:
- Viscosity: 1e-5
- Number of samples: 1200
- Final time: 20
- Resolution: 64x64
- Time steps recorded: 20
"""

import sys
sys.path.append('../')

import torch
import math
import scipy.io
from timeit import default_timer
from tqdm import tqdm

# Import from the data generation module
from data.data_generation import GaussianRF, navier_stokes_2d

def main():
    # Parameters
    resolution = 64
    N = 1200  # Number of samples
    visc = 1e-5  # Viscosity
    T_final = 20.0  # Final time
    delta_t = 1e-4  # Time step
    record_steps = 20  # Number of time snapshots to save
    batch_size = 20  # Process in batches
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set up Gaussian Random Field for initial conditions
    GRF = GaussianRF(2, resolution, alpha=2.5, tau=7, device=device)
    
    # Create spatial grid
    t = torch.linspace(0, 1, resolution+1, device=device)
    t = t[0:-1]
    
    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    # Initialize arrays to store data
    a = torch.zeros(N, resolution, resolution)  # Initial conditions
    u = torch.zeros(N, resolution, resolution, record_steps)  # Solutions
    
    # Generate data in batches
    c = 0
    t0 = default_timer()
    
    print(f"\nGenerating {N} samples...")
    print(f"Parameters: resolution={resolution}, visc={visc}, T={T_final}, steps={record_steps}")
    print(f"Batch size: {batch_size}\n")
    
    for j in tqdm(range(N // batch_size), desc="Progress"):
        # Sample random initial conditions
        w0 = GRF.sample(batch_size)
        
        # Solve Navier-Stokes equation
        sol, sol_t = navier_stokes_2d(w0, f, visc, T_final, delta_t, record_steps)
        
        # Store results
        a[c:(c+batch_size), ...] = w0
        u[c:(c+batch_size), ...] = sol
        
        c += batch_size
        
        # Print progress every 10 batches
        if (j + 1) % 10 == 0:
            t1 = default_timer()
            elapsed = t1 - t0
            avg_time = elapsed / (j + 1)
            remaining = avg_time * (N // batch_size - j - 1)
            print(f"\nBatch {j+1}/{N//batch_size} | Samples: {c}/{N} | "
                  f"Elapsed: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s")
    
    t1 = default_timer()
    print(f"\n✓ Data generation complete! Total time: {t1-t0:.1f} seconds")
    
    # Save to MATLAB file
    output_file = 'NavierStokes_V1e-5_N1200_T20.mat'
    print(f"\nSaving to {output_file}...")
    
    scipy.io.savemat(
        output_file,
        mdict={
            'a': a.cpu().numpy(),
            'u': u.cpu().numpy(),
            't': sol_t.cpu().numpy()
        }
    )
    
    print(f"✓ File saved successfully!")
    print(f"\nData shapes:")
    print(f"  Initial conditions (a): {a.shape}")
    print(f"  Solutions (u): {u.shape}")
    print(f"  Time points (t): {sol_t.shape}")

if __name__ == "__main__":
    main()




