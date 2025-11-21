# Experiments Notebook - Setup Guide

## ✅ What I've Fixed

The `experiments.ipynb` notebook has been updated to work with the current codebase:

### Changes Made:
1. **Added path setup cell** (Cell 2): Adds parent directory to Python path
2. **Updated imports** (Cell 3): Changed from non-existent modules to actual project structure:
   - `FNO.PyTorch.fno.FNO` instead of `FNO.fno_2d_time.FNO2DTime`
   - `losses.lploss.LpLoss` instead of `FNO.lploss.LpLoss`
   - `training.train.train_model` instead of `FNO.train.train_model`
3. **Created FNO2DTime wrapper** (Cell 4): A compatibility wrapper that makes the existing `FNO` class work with the notebook's expected interface

## 🚧 What Still Needs to Be Done

### Data Generation Required
The notebook still needs the data file: `NavierStokes_V1e-5_N1200_T20.mat`

#### Option 1: Quick Test (Recommended) - ~30-40 minutes
```bash
cd /Users/genghisluo/Downloads/Fourier-Neural-Operator-main/notebooks
source ../venv/bin/activate
python generate_small_data.py
```

This generates 100 samples (instead of 1200). Then adjust data loading in the notebook:
```python
# In the data loading cell, change:
ntrain = 80   # instead of 1000
ntest = 20    # instead of 200
```

#### Option 2: Full Dataset - ~4-5 hours
```bash
cd /Users/genghisluo/Downloads/Fourier-Neural-Operator-main/notebooks
source ../venv/bin/activate
nohup python generate_navier_stokes_data.py > data_gen.log 2>&1 &

# Monitor progress:
tail -f data_gen.log
```

## 📝 How to Use the Notebook

### 1. Make sure virtual environment is activated:
```bash
source /Users/genghisluo/Downloads/Fourier-Neural-Operator-main/venv/bin/activate
```

### 2. Start Jupyter:
```bash
cd /Users/genghisluo/Downloads/Fourier-Neural-Operator-main/notebooks
jupyter notebook experiments.ipynb
```

### 3. Run cells in order:
- Cell 0-1: Markdown/headers
- Cell 2: ✅ Path setup (NEW - run this!)
- Cell 3: ✅ Imports (FIXED - run this!)
- Cell 4: ✅ FNO2DTime wrapper (NEW - run this!)
- Cell 5+: Data loading and experiments (need data file first!)

## 🔧 Troubleshooting

### "No module named 'FNO'"
- Make sure you ran Cell 2 (path setup)
- Make sure you're in the notebooks/ directory when running Jupyter

### "FileNotFoundError: NavierStokes_V1e-5_N1200_T20.mat"
- You need to generate the data first (see "Data Generation Required" above)

### "CUDA out of memory" 
- Change `.cuda()` to `.cpu()` in model creation cells
- Or reduce batch size

### Import errors with utilities
- The `utilities/utils.py` file exists and should work
- If issues persist, check that you're running from within the notebooks/ directory

## 📊 Expected Results

Once data is generated, the notebook will:
1. Test different FNO architectures with varying modes
2. Compare training with different data amounts
3. Test different spatial resolutions  
4. Benchmark and compare all variants

## 💡 Tips

- **Start small**: Use the quick test dataset (100 samples) to verify everything works
- **Then scale up**: Generate full dataset overnight if needed
- **GPU recommended**: Training will be much faster with CUDA
- **Check RAM**: Make sure you have enough memory for the dataset

## 🆘 Still Having Issues?

Check:
1. Virtual environment is activated
2. All packages installed: `pip list` should show torch, matplotlib, scipy, etc.
3. You're in the correct directory: `/Users/genghisluo/Downloads/Fourier-Neural-Operator-main/notebooks/`
4. Python version: Should be 3.9+ (JAX removed from requirements due to 3.10+ requirement)




