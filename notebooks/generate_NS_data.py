#!/usr/bin/env python3
"""
Generate high-resolution Navier-Stokes data for super-resolution testing.

This script generates data for the 2D Navier-Stokes equation with:
- Viscosity: 1e-5
- Number of samples: customizable (default: 100)
- Final time: 20
- Resolution: 256x256 or 512x512
- Time steps recorded: 20
- Output format: .mat file (MATLAB compatible, smaller file size)

Usage:
    python generate_NS_data.py --resolution 256 --samples 100
    python generate_NS_data.py --resolution 512 --samples 20
"""

import sys
sys.path.append('../')

import torch
import math
import numpy as np
import scipy.io
import argparse
from timeit import default_timer
from tqdm import tqdm

# Import from the data generation module
from data.data_generation import GaussianRF, navier_stokes_2d


def main():
    parser = argparse.ArgumentParser(description='Generate high-res NS data')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution (256 or 512)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--visc', type=float, default=1e-5, help='Viscosity')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    args = parser.parse_args()
    
    # Parameters
    resolution = args.resolution
    N = args.samples
    visc = args.visc
    T_final = 20.0
    delta_t = 1e-4
    record_steps = 20
    batch_size = min(N, 5)  # Smaller batch for high-res
    
    # Output filename - now .mat instead of .npy
    if args.output:
        output_file = 'data/' + args.output
        # Ensure .mat extension
        if not output_file.endswith('.mat'):
            output_file = output_file.rsplit('.', 1)[0] + '.mat'
    else:
        output_file = f'data/ns_{resolution}x{resolution}_{N}_v{visc:.0e}.mat'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n{'='*50}")
    print(f"Generating {N} samples at {resolution}x{resolution} resolution")
    print(f"Parameters: visc={visc}, T={T_final}, steps={record_steps}")
    print(f"{'='*50}\n")
    
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
    
    for j in tqdm(range(N // batch_size), desc="Generating"):
        # Sample random initial conditions
        w0 = GRF.sample(batch_size)
        
        # Solve Navier-Stokes equation
        sol, sol_t = navier_stokes_2d(w0, f, visc, T_final, delta_t, record_steps)
        
        # Store results
        a[c:(c+batch_size), ...] = w0  # Save initial conditions
        u[c:(c+batch_size), ...] = sol
        c += batch_size
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    t1 = default_timer()
    print(f"\n✓ Data generation complete! Total time: {t1-t0:.1f} seconds")
    print(f"  Average time per batch: {(t1-t0)/(N//batch_size):.2f} seconds")
    
    # Save to MATLAB .mat file (more compact than .npy)
    print(f"\nSaving to {output_file}...")
    scipy.io.savemat(
        output_file,
        mdict={
            'a': a.cpu().numpy(),      # Initial conditions [N, H, W]
            'u': u.cpu().numpy(),      # Solutions [N, H, W, T]
            't': sol_t.cpu().numpy()   # Time points [T]
        }
    )
    
    print(f"✓ File saved successfully!")
    print(f"\nData shapes:")
    print(f"  Initial conditions (a): {a.shape} - {N} samples at {resolution}x{resolution}")
    print(f"  Solutions (u): {u.shape} - {N} samples, {resolution}x{resolution}, {record_steps} time steps")
    print(f"  Time points (t): {sol_t.shape}")
    
    # Estimate file size
    import os
    if os.path.exists(output_file):
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nFile size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()

