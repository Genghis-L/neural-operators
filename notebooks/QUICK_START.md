# Quick Start - Multi-GPU NS Data Generation

## Resource Requirements

### For 4× GPUs (A100 Recommended)

```bash
#SBATCH --time=24:00:00       # 24 hours
#SBATCH --mem=300G            # 300 GB RAM
#SBATCH --cpus-per-task=16    # 16 CPU cores
#SBATCH --gres=gpu:a100:4     # 4× NVIDIA A100 GPUs
```

### Time Estimates

| GPUs | Total Time | Speedup |
|------|------------|---------|
| 1 GPU | ~58 hours | 1.0× |
| 2 GPUs | ~32 hours | 1.8× |
| **4 GPUs** | **~18 hours** | **3.2×** |
| 8 GPUs | ~12 hours | 4.8× (diminishing returns) |

### Recommended GPU Types

**Best to Worst:**
1. **NVIDIA A100** (40GB or 80GB) - Optimal, high memory bandwidth
2. **NVIDIA H100** - Faster but expensive, overkill
3. **NVIDIA A40/A30** - Good alternative
4. **NVIDIA V100** - Works but slower memory bandwidth
5. **RTX 4090/3090** - Consumer cards, less stable for long runs

**Any NVIDIA GPU with CUDA support works**, but A100 gives best performance/cost.

---

## Usage

### Run All Resolutions (64, 128, 256, 512, 1024)

```bash
cd /Users/genghisluo/Desktop/neural-operators/notebooks

# 4 GPUs
for res in 64 128 256 512 1024; do
    python3 generate_NS_data_multigpu.py --resolution $res --samples 1200 --visc 1e-5 --gpus 0,1,2,3
done
```

### Run Single Resolution

```bash
# Example: 512×512 with 4 GPUs
python3 generate_NS_data_multigpu.py --resolution 512 --samples 1200 --visc 1e-5 --gpus 0,1,2,3
```

### Adjust for Your GPU Count

```bash
# 2 GPUs
--gpus 0,1

# 4 GPUs
--gpus 0,1,2,3

# 8 GPUs
--gpus 0,1,2,3,4,5,6,7
```

---

## SLURM Job Script

Create `run_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ns_gen
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --output=logs/ns_%j.out

cd notebooks

for res in 64 128 256 512 1024; do
    python3 generate_NS_data_multigpu.py \
        --resolution $res \
        --samples 1200 \
        --visc 1e-5 \
        --gpus 0,1,2,3
    echo "✓ ${res}x${res} complete"
done
```

Submit:
```bash
sbatch run_job.sh
```

---

## How It Works

**With 4 GPUs and 1200 samples:**
- GPU 0: samples 0-299
- GPU 1: samples 300-599
- GPU 2: samples 600-899
- GPU 3: samples 900-1199

Each GPU works independently, results combined automatically.

---

## Monitoring

```bash
# Check GPU usage (should be 90-100%)
watch -n 1 nvidia-smi

# Check generated files
ls -lh data/*.mat

# Check SLURM job
squeue -u $USER
```

---

## Expected Output

```
data/ns_64x64_1200_v1e-05.mat       (~50 MB)
data/ns_128x128_1200_v1e-05.mat     (~150 MB)
data/ns_256x256_1200_v1e-05.mat     (~500 MB)
data/ns_512x512_1200_v1e-05.mat     (~2 GB)
data/ns_1024x1024_1200_v1e-05.mat   (~8 GB)

Total: ~11 GB
```

---

## Troubleshooting

**Out of memory:** Reduce batch size
```bash
--batch-size 1
```

**CUDA error:** Check available GPUs
```bash
nvidia-smi
```

**Slow performance:** Verify GPU utilization is high (~95%)

---

## Full Resource Table

| Configuration | Time | RAM | CPU | GPU Type | Best Use |
|--------------|------|-----|-----|----------|----------|
| 1× A100 | 58h | 300GB | 16 | A100 40/80GB | Budget |
| 2× A100 | 32h | 300GB | 16 | A100 40/80GB | Balanced |
| **4× A100** | **18h** | **300GB** | **16** | **A100 40/80GB** | **Recommended** |
| 8× A100 | 12h | 300GB | 16 | A100 40/80GB | Fast turnaround |

**Your setup (4 GPUs):** 18 hours for all data generation.
