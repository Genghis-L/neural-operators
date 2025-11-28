#!/usr/bin/env python3
"""
Multi-GPU Navier-Stokes data generation.
Splits sample generation across multiple GPUs for faster processing.
Outputs data in PyTorch (.pt) format as partitioned files for scalability.

Usage:
    python generate_NS_data_multigpu.py --resolution 512 --samples 1200 --visc 1e-5 --gpus 0,1,2,3
"""

import sys
sys.path.append('../')

import torch
import torch.multiprocessing as mp
import math
import argparse
import os
from pathlib import Path
from timeit import default_timer
from tqdm import tqdm

from data.data_generation import GaussianRF, navier_stokes_2d


def generate_batch_on_gpu(gpu_id, resolution, samples_start, samples_end, visc, T_final, 
                          delta_t, record_steps, batch_size, temp_dir, output_queue):
    """
    Generate a batch of NS data on a specific GPU.
    Saves once after all batches complete to maximize GPU utilization.
    
    Args:
        gpu_id: GPU device ID
        resolution: Grid resolution
        samples_start: Starting sample index
        samples_end: Ending sample index
        visc: Viscosity
        T_final: Final time
        delta_t: Time step
        record_steps: Number of time steps to record
        batch_size: Batch size for processing
        temp_dir: Directory for temporary files
        output_queue: Queue to store completion status
    """
    # Set device
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    n_samples = samples_end - samples_start
    
    print(f"GPU {gpu_id}: Generating samples {samples_start} to {samples_end} ({n_samples} samples)")
    
    # Set up Gaussian Random Field
    GRF = GaussianRF(2, resolution, alpha=2.5, tau=7, device=device)
    
    # Create spatial grid and forcing
    t = torch.linspace(0, 1, resolution+1, device=device)[:-1]
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1 * (torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    # Initialize arrays
    a_local = torch.zeros(n_samples, resolution, resolution)
    u_local = torch.zeros(n_samples, resolution, resolution, record_steps)
    
    # Generate data in batches
    c = 0
    t0 = default_timer()
    n_batches = math.ceil(n_samples / batch_size)
    
    for j in tqdm(range(n_batches), desc=f"GPU {gpu_id}", position=gpu_id):
        current_batch_size = min(batch_size, n_samples - c)
        
        # Sample initial conditions
        w0 = GRF.sample(current_batch_size)
        
        # Solve NS equation
        sol, sol_t = navier_stokes_2d(w0, f, visc, T_final, delta_t, record_steps)
        
        # Store results
        a_local[c:(c+current_batch_size), ...] = w0.cpu()
        u_local[c:(c+current_batch_size), ...] = sol.cpu()
        c += current_batch_size
    
    t1 = default_timer()
    print(f"GPU {gpu_id}: Complete in {t1-t0:.1f}s | Avg: {(t1-t0)/n_batches:.2f}s/batch")
    
    # Save checkpoint after all batches are complete
    checkpoint_file = os.path.join(temp_dir, f'gpu{gpu_id}_completed.pt')
    torch.save(
        {
            'a': a_local,
            'u': u_local,
            't': sol_t.cpu(),
            'samples_start': samples_start,
            'samples_end': samples_end,
            'total_samples': n_samples
        },
        checkpoint_file
    )
    
    # Put completion status in queue
    output_queue.put({
        'gpu_id': gpu_id,
        'samples_start': samples_start,
        'samples_end': samples_end,
        'checkpoint_file': checkpoint_file,
        'success': True
    })


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU NS data generation')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution')
    parser.add_argument('--samples', type=int, default=1200, help='Total samples')
    parser.add_argument('--visc', type=float, default=1e-5, help='Viscosity')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='GPU IDs (comma-separated, e.g., 0,1,2,3)')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size per GPU')
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    n_gpus = len(gpu_ids)
    
    # Parameters
    resolution = args.resolution
    N = args.samples
    visc = args.visc
    T_final = 60.0
    delta_t = 1e-4
    record_steps = 100
    
    # Auto-set batch size based on resolution
    if args.batch_size is None:
        if resolution <= 64:
            batch_size = 1000 # Aggressive batching for small resolution to fight CPU bottleneck
        elif resolution <= 256:
            batch_size = 200  # Increased for A100
        elif resolution <= 512:
            batch_size = 100  # Increased for A100
        else:  # 1024+
            batch_size = 20
    else:
        batch_size = args.batch_size
    
    # Output base name (no extension, will be partitioned)
    if args.output:
        base_output = 'data/' + args.output
        # Strip any extension
        if base_output.endswith('.mat') or base_output.endswith('.pt'):
            base_output = base_output.rsplit('.', 1)[0]
    else:
        base_output = f'data/#{N}_ns_{resolution}x{resolution}_v{visc:.0e}_T={T_final}_steps={record_steps}'
    
    # Create temporary directory for incremental saves
    temp_dir = f'data/temp_res{resolution}_n{N}'
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Multi-GPU NS Data Generation (PyTorch Format)")
    print(f"{'='*60}")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Samples: {N}")
    print(f"Viscosity: {visc}")
    print(f"GPUs: {gpu_ids}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Output format: PyTorch (.pt) partitioned files")
    print(f"Temp directory: {temp_dir}")
    print(f"{'='*60}\n")
    
    # Check GPU availability
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_id} not available. Found {torch.cuda.device_count()} GPUs.")
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    print()
    
    # Distribute samples across GPUs
    samples_per_gpu = N // n_gpus
    sample_ranges = []
    for i, gpu_id in enumerate(gpu_ids):
        start = i * samples_per_gpu
        end = start + samples_per_gpu if i < n_gpus - 1 else N  # Last GPU takes remainder
        sample_ranges.append((gpu_id, start, end))
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    output_queue = mp.Queue()
    processes = []
    
    # Start processes
    t0 = default_timer()
    for gpu_id, start, end in sample_ranges:
        p = mp.Process(
            target=generate_batch_on_gpu,
            args=(gpu_id, resolution, start, end, visc, T_final, 
                  delta_t, record_steps, batch_size, temp_dir, output_queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    t1 = default_timer()
    
    # Collect results
    print("\nCollecting results from all GPUs...")
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    
    # Sort by sample index
    results.sort(key=lambda x: x['samples_start'])
    
    # Check all GPUs completed successfully
    failed_gpus = [r['gpu_id'] for r in results if not r.get('success', False)]
    if failed_gpus:
        print(f"⚠ Warning: GPUs {failed_gpus} did not complete successfully")
    
    # Move/Rename partitioned files to final location
    print(f"\nFinalizing output files...")
    
    # Move/Rename files instead of merging into RAM
    for result in results:
        gpu_id = result['gpu_id']
        checkpoint_file = result['checkpoint_file']
        
        if os.path.exists(checkpoint_file):
            # New filename: base_partX.pt
            final_part_file = f"{base_output}_part{gpu_id}.pt"
            
            print(f"  Moving GPU {gpu_id} output -> {final_part_file}")
            os.rename(checkpoint_file, final_part_file)
            
            # Log size
            size_gb = os.path.getsize(final_part_file) / (1024**3)
            print(f"  ✓ Part {gpu_id}: {size_gb:.2f} GB")
        else:
            print(f"  ✗ Missing checkpoint file for GPU {gpu_id}")

    print(f"\n{'='*60}")
    print(f"✓ Data generation complete!")
    print(f"{'='*60}")
    print(f"Output files: {base_output}_part*.pt")
    print(f"Total parts: {len(results)}")
    print(f"Total time: {t1-t0:.1f}s")
    print(f"Speedup: ~{len(gpu_ids)}x (using {len(gpu_ids)} GPUs)")
    print(f"{'='*60}\n")
    
    # Clean up temporary directory
    print(f"\nCleaning up temporary directory...")
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✓ Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"⚠ Could not remove temp directory: {e}")


if __name__ == "__main__":
    main()

