#!/usr/bin/env python3
"""
Generate a SMALL Navier-Stokes dataset for testing: NavierStokes_V1e-5_N1200_T20.mat
This will only generate 100 samples instead of 1200 for quick testing.
"""

import sys
sys.path.append('../')

import torch
import math
import scipy.io
from timeit import default_timer
from tqdm import tqdm

from data.data_generation import GaussianRF, navier_stokes_2d

def main():
    # Reduced parameters for quick testing
    resolution = 64
    N = 100  # Only 100 samples instead of 1200
    visc = 1e-5
    T_final = 20.0
    delta_t = 1e-4
    record_steps = 20
    batch_size = 10  # Smaller batches
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"⚠️  WARNING: Generating only {N} samples for testing (not 1200)")
    
    GRF = GaussianRF(2, resolution, alpha=2.5, tau=7, device=device)
    
    t = torch.linspace(0, 1, resolution+1, device=device)
    t = t[0:-1]
    
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    a = torch.zeros(N, resolution, resolution)
    u = torch.zeros(N, resolution, resolution, record_steps)
    
    c = 0
    t0 = default_timer()
    
    print(f"\nGenerating {N} samples (this will take ~30-40 minutes on CPU)...")
    
    for j in tqdm(range(N // batch_size), desc="Progress"):
        w0 = GRF.sample(batch_size)
        sol, sol_t = navier_stokes_2d(w0, f, visc, T_final, delta_t, record_steps)
        
        a[c:(c+batch_size), ...] = w0
        u[c:(c+batch_size), ...] = sol
        c += batch_size
    
    t1 = default_timer()
    print(f"\n✓ Complete! Time: {(t1-t0)/60:.1f} minutes")
    
    # Save with the same filename so the notebook works
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
    
    print(f"✓ Saved! (Note: only {N} samples, not 1200)")
    print(f"  You can use first 80 for training, 20 for testing")

if __name__ == "__main__":
    main()




