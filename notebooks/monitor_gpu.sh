#!/bin/bash
# GPU监控脚本 - 简洁版，只记录日志
# 使用方法: ./monitor_gpu.sh [interval_seconds] [log_file]

INTERVAL=${1:-10}  # 默认每10秒更新一次
LOG_FILE=${2:-"gpu_monitor_$(date +%Y%m%d_%H%M%S).log"}

echo "GPU监控启动 - 间隔: ${INTERVAL}秒 - 日志: ${LOG_FILE}"

# 写入日志头
{
    echo "GPU Monitor Started: $(date)"
    echo "Update Interval: ${INTERVAL} seconds"
    echo "Format: [Timestamp] GPU_ID, Util%, Mem%, MemUsed_MB, Temp_C, Power_W"
    echo "----------------------------------------"
} > "$LOG_FILE"

# 后台运行监控
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 记录到日志 - 简洁格式
    echo "[$TIMESTAMP]" >> "$LOG_FILE"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw \
        --format=csv,noheader,nounits >> "$LOG_FILE"
    
    sleep "$INTERVAL"
done

