# Data Generation Guide

## Problem
The notebook requires `NavierStokes_V1e-5_N1200_T20.mat` which doesn't exist and needs to be generated.

## Why it's slow
- Generating 1200 high-resolution PDE simulations
- Running on CPU (no GPU acceleration)  
- Each batch takes ~4+ minutes
- **Total estimated time: 4-5 hours**

## Solutions

### Option 1: Quick Test Dataset (RECOMMENDED)
Generate only 100 samples for testing (~30-40 minutes):

```bash
cd notebooks
source ../venv/bin/activate
python generate_small_data.py
```

Then in the notebook, adjust the training size:
```python
ntrain = 80   # instead of 1000
ntest = 20    # instead of 200
```

### Option 2: Full Dataset Generation
If you really need 1200 samples, run in background:

```bash
cd notebooks
source ../venv/bin/activate
nohup python generate_navier_stokes_data.py > data_generation.log 2>&1 &
```

Check progress:
```bash
tail -f data_generation.log
```

### Option 3: Try to Download Pre-generated Data
The original FNO paper authors may have shared data. Search for:
- https://github.com/zongyi-li/fourier_neural_operator
- Look in their data/ directory or releases

### Option 4: Use a Different Example
Try the Heat Equation notebook instead, which might be faster:
```
Heat_Equation2D_FNO.ipynb
```

## What the data contains
- **a**: Initial vorticity fields (N, 64, 64)
- **u**: Solutions over time (N, 64, 64, 20)  
- **t**: Time points (20,)

Where N is the number of samples (1200 for full, 100 for test)

## Notebook adjustments needed for small dataset
If using 100 samples instead of 1200:

```python
# Original
ntrain = 1000
ntest = 200

# Change to
ntrain = 80
ntest = 20
```

