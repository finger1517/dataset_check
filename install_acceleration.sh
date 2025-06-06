#!/bin/bash
# 安装视频处理加速依赖

echo "🚀 安装视频处理加速依赖"
echo "=========================="

# 基础依赖
echo "📦 安装基础依赖..."
pip install opencv-python numpy

# Numba JIT编译
echo "⚡ 安装Numba (JIT编译)..."
pip install numba

# CuPy (GPU加速) - 需要CUDA
echo "🎮 尝试安装CuPy (GPU加速)..."
pip install cupy-cuda11x || pip install cupy-cuda12x || echo "⚠️ CuPy安装失败，将使用CPU"

# FFmpeg-python (更快的视频解码)
echo "🎬 安装FFmpeg-python..."
pip install ffmpeg-python

# 检查安装状态
echo ""
echo "🔍 检查安装状态:"
python3 -c "
try:
    import numba
    print('✅ Numba: 已安装')
except ImportError:
    print('❌ Numba: 未安装')

try:
    import cupy
    print('✅ CuPy: 已安装')
except ImportError:
    print('❌ CuPy: 未安装')

try:
    import ffmpeg
    print('✅ FFmpeg-python: 已安装')
except ImportError:
    print('❌ FFmpeg-python: 未安装')
"

echo ""
echo "✅ 安装完成！运行 python accelerated_video_processing.py 进行测试" 