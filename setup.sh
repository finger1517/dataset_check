#!/bin/bash
# 视频数据集处理效能比较工具安装脚本

echo "🎬 视频数据集处理效能比较工具 - 环境设置"
echo "================================================"

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python版本检查通过: $(python3 --version)"
else
    echo "❌ Python版本过低，需要Python 3.8或更高版本"
    echo "当前版本: $(python3 --version)"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未找到，请先安装pip"
    exit 1
fi

echo "✅ pip检查通过"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv video_dataset_env
    source video_dataset_env/bin/activate
    echo "✅ 虚拟环境已创建并激活"
fi

# 升级pip
echo "⬆️ 升级pip..."
pip3 install --upgrade pip

# 安装依赖包
echo "📥 安装依赖包..."
pip3 install -r requirements.txt

# 检查关键包是否安装成功
echo "🔍 检查安装状态..."

packages=("torch" "torchvision" "ray" "datasets" "opencv-python" "numpy" "pandas")
all_installed=true

for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package 安装成功"
    else
        echo "❌ $package 安装失败"
        all_installed=false
    fi
done

if [ "$all_installed" = true ]; then
    echo ""
    echo "🎉 所有依赖包安装成功！"
    echo ""
    echo "使用方法："
    echo "1. 运行完整测试: python3 video_dataset_comparison.py"
    echo "2. 运行交互式测试: python3 run_benchmark.py"
    echo "3. 单独测试各框架:"
    echo "   - PyTorch: python3 pytorch_data.py"
    echo "   - Ray: python3 ray_data.py"
    echo "   - HuggingFace: python3 huggingface_data.py"
    echo ""
    echo "📖 详细说明请查看 README.md"
else
    echo ""
    echo "❌ 部分依赖包安装失败，请检查错误信息并手动安装"
    exit 1
fi

# 检查视频数据目录
if [ -d "../video_data" ]; then
    video_count=$(find ../video_data -name "*.mp4" | wc -l)
    echo "📁 找到视频数据目录，包含 $video_count 个MP4文件"
else
    echo "⚠️ 未找到视频数据目录 (../video_data)"
    echo "请确保视频数据目录存在并包含MP4文件"
fi

echo ""
echo "🚀 环境设置完成！现在可以开始使用工具了。" 