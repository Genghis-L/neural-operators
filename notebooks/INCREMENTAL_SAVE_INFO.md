# Multi-GPU NS Data Generation - Incremental Save Mode

## ✨ 新功能

修改后的 `generate_NS_data_multigpu.py` 现在支持**增量保存**，防止因作业被终止而丢失数据。

## 🔧 工作原理

1. **每个batch立即保存**: 每个GPU每完成一个batch（1/25进度）就立即保存到临时文件
2. **自动合并**: 所有GPU完成后，自动合并所有临时文件到最终的 `.mat` 文件
3. **自动清理**: 合并完成后自动删除临时文件
4. **防止数据丢失**: 即使作业被中断，已完成的batch数据也已保存

## 📁 文件结构

运行时会创建临时目录：
```
data/
├── temp_res64_n1000/           # 临时目录
│   ├── gpu0_progress.mat       # GPU 0的进度（实时更新）
│   ├── gpu1_progress.mat       # GPU 1的进度（实时更新）
│   ├── gpu2_progress.mat       # GPU 2的进度（实时更新）
│   └── gpu3_progress.mat       # GPU 3的进度（实时更新）
└── #1000_ns_64x64_v1e-05_...mat  # 最终合并的文件
```

**注意**: 临时目录在最终文件保存后会自动删除。

## 🚀 使用方法

### 方法1: 在Jupyter Notebook中运行（推荐）

```python
# 在单独的cell中运行每个分辨率
!cd /scratch/kl4747/Fourier-Neural-Operator-main/notebooks && \
 python generate_NS_data_multigpu.py --resolution 64 --samples 1000 --visc 1e-5 --gpus 0,1,2,3
```

等第一个完成并确认文件保存后，再运行下一个分辨率。

### 方法2: 使用bash脚本

```bash
#!/bin/bash
cd /scratch/kl4747/Fourier-Neural-Operator-main/notebooks

for res in 64 128 256 512 1024; do
    echo "Starting resolution ${res}x${res}..."
    python generate_NS_data_multigpu.py \
        --resolution $res \
        --samples 1000 \
        --visc 1e-5 \
        --gpus 0,1,2,3
    
    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ Resolution $res completed successfully"
    else
        echo "✗ Resolution $res failed"
        exit 1
    fi
done
```

## 📊 输出示例

```
============================================================
Multi-GPU NS Data Generation (Incremental Save Mode)
============================================================
Resolution: 64x64
Samples: 1000
Viscosity: 1e-05
GPUs: [0, 1, 2, 3]
Batch size per GPU: 10
Temp directory: data/temp_res64_n1000
============================================================

GPU 0: NVIDIA A100-SXM4-80GB
  Memory: 85.0 GB
...

GPU 0:   4%|▍  | 1/25 [03:24<1:21:48, 204.52s/it]
  → 自动保存到 data/temp_res64_n1000/gpu0_progress.mat

GPU 0:   8%|▊  | 2/25 [06:48<1:18:17, 204.24s/it]
  → 自动更新 data/temp_res64_n1000/gpu0_progress.mat

...

Merging checkpoint files...
  ✓ Loaded GPU 0: samples 0-250
  ✓ Loaded GPU 1: samples 250-500
  ✓ Loaded GPU 2: samples 500-750
  ✓ Loaded GPU 3: samples 750-1000

✓ Data generation complete!
Saving to data/#1000_ns_64x64_v1e-05_T=60.0_steps=2000.mat...
✓ File saved successfully!

Cleaning up temporary files...
✓ Removed temporary directory: data/temp_res64_n1000
```

## 🛡️ 容错机制

如果作业被中断：
1. 已完成的batch数据已保存在 `data/temp_res*_n*/gpu*_progress.mat`
2. 你可以手动检查这些文件来恢复部分数据
3. 或者重新运行脚本（会覆盖临时文件，从头开始）

## 💾 磁盘空间考虑

- **临时空间**: 需要约 2x 最终文件大小（每个GPU一个临时文件 + 最终合并文件）
- **峰值使用**: 在合并阶段会同时存在临时文件和最终文件
- **自动清理**: 合并完成后临时文件会被删除

## 示例文件大小（仅供参考）

| Resolution | Samples | Final .mat size | Temp space needed |
|-----------|---------|-----------------|-------------------|
| 64x64     | 1000    | ~150 MB         | ~300 MB           |
| 128x128   | 1000    | ~600 MB         | ~1.2 GB           |
| 256x256   | 1000    | ~2.5 GB         | ~5 GB             |
| 512x512   | 1000    | ~10 GB          | ~20 GB            |
| 1024x1024 | 1000    | ~40 GB          | ~80 GB            |

## ⚙️ 参数说明

```bash
python generate_NS_data_multigpu.py \
    --resolution 64      # 网格分辨率
    --samples 1000       # 总样本数
    --visc 1e-5          # 粘度系数
    --gpus 0,1,2,3       # 使用的GPU ID（逗号分隔）
    --batch-size 10      # 每个GPU的batch size（可选，会自动设置）
    --output custom.mat  # 自定义输出文件名（可选）
```

## 🐛 故障排查

### 问题: 临时目录没有被删除
**原因**: 脚本可能在清理之前被中断  
**解决**: 手动删除 `data/temp_res*_n*` 目录

### 问题: 作业仍然被终止
**原因**: GPU在合并阶段空闲时间过长  
**解决**: 
1. 使用更小的batch size以减少总运行时间
2. 向作业调度器请求更长的GPU空闲容忍时间
3. 手动合并临时文件（见下方）

### 手动合并临时文件（紧急情况）

```python
import scipy.io
import numpy as np

resolution = 64
N = 1000
record_steps = 2000
temp_dir = 'data/temp_res64_n1000'

a = np.zeros((N, resolution, resolution))
u = np.zeros((N, resolution, resolution, record_steps))

for gpu_id in [0, 1, 2, 3]:
    data = scipy.io.loadmat(f'{temp_dir}/gpu{gpu_id}_progress.mat')
    start = int(data['samples_start'][0, 0])
    end = int(data['samples_end'][0, 0])
    a[start:end] = data['a']
    u[start:end] = data['u']
    sol_t = data['t']

scipy.io.savemat('data/recovered_data.mat', {'a': a, 'u': u, 't': sol_t})
```

## ✅ 验证数据完整性

运行完成后，验证数据：

```python
import scipy.io

data = scipy.io.loadmat('data/#1000_ns_64x64_v1e-05_T=60.0_steps=2000.mat')
print(f"Initial conditions shape: {data['a'].shape}")  # 应该是 (1000, 64, 64)
print(f"Solutions shape: {data['u'].shape}")           # 应该是 (1000, 64, 64, 2000)
print(f"Time points shape: {data['t'].shape}")         # 应该是 (2000,)
```

