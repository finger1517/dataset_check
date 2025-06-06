# 视频数据集处理效能比较工具

这个工具包提供了使用Ray、HuggingFace和PyTorch三种不同数据集框架处理视频数据的实现，并能够比较它们的性能表现。

## 功能特性

- 🎬 **视频帧提取**: 从MP4视频中均匀提取16帧
- 🖼️ **图像预处理**: 将帧调整为224x224尺寸并进行标准化
- ⚡ **多框架支持**: Ray Data、HuggingFace Datasets、PyTorch Dataset
- 📊 **性能比较**: 详细的处理速度和内存使用情况分析
- 🔧 **多种优化**: 预加载、多进程、缓存等优化策略

## 文件结构

```
dataset_maker/
├── video_dataset_comparison.py    # 主要的比较脚本
├── ray_data.py                   # Ray数据集专门实现
├── huggingface_data.py          # HuggingFace数据集专门实现
├── pytorch_data.py              # PyTorch数据集专门实现
├── requirements.txt             # 依赖包列表
└── README.md                    # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch>=1.12.0` - PyTorch深度学习框架
- `torchvision>=0.13.0` - PyTorch视觉库
- `ray[data]>=2.0.0` - Ray分布式计算框架
- `datasets>=2.0.0` - HuggingFace数据集库
- `opencv-python>=4.6.0` - OpenCV视频处理
- `numpy>=1.21.0` - 数值计算
- `pandas>=1.4.0` - 数据分析
- `psutil>=5.9.0` - 系统监控

## 使用方法

### 1. 运行完整比较测试

```bash
cd dataset_maker
python video_dataset_comparison.py
```

这将：
- 自动搜索 `../video_data` 目录中的所有MP4文件
- 使用三种数据集框架分别处理视频数据
- 测量处理时间、内存使用等性能指标
- 生成详细的比较报告
- 将结果保存到CSV文件

### 2. 单独测试各个框架

#### Ray数据集测试
```bash
python ray_data.py
```

#### HuggingFace数据集测试
```bash
python huggingface_data.py
```

#### PyTorch数据集测试
```bash
python pytorch_data.py
```

## 数据集实现详情

### Ray数据集 (`ray_data.py`)
- **特点**: 分布式处理，适合大规模数据
- **优势**: 自动并行化，内存高效
- **配置**: 可调整批处理大小和工作进程数

```python
from ray_data import RayVideoDataset

dataset = RayVideoDataset(video_paths)
ray_dataset = dataset.create_dataset(batch_size=4)
```

### HuggingFace数据集 (`huggingface_data.py`)
- **特点**: 强大的缓存和序列化功能
- **优势**: 数据持久化，多进程处理
- **配置**: 支持缓存到磁盘，可配置处理进程数

```python
from huggingface_data import HuggingFaceVideoDataset

dataset = HuggingFaceVideoDataset(video_paths)
hf_dataset = dataset.create_dataset(num_proc=4)
```

### PyTorch数据集 (`pytorch_data.py`)
- **特点**: 深度学习生态系统集成度高
- **优势**: 多种优化策略（预加载、可迭代等）
- **配置**: 支持多工作进程、GPU加速

```python
from pytorch_data import PyTorchVideoDataset

dataset = PyTorchVideoDataset(video_paths, preload=True)
dataloader = DataLoader(dataset, batch_size=4, num_workers=2)
```

## 性能优化策略

### Ray数据集优化
- 调整 `batch_size` 参数
- 使用 `repartition()` 优化数据分布
- 配置合适的CPU资源分配

### HuggingFace数据集优化
- 启用磁盘缓存减少重复处理
- 使用多进程 (`num_proc`) 加速处理
- 利用Arrow格式的高效存储

### PyTorch数据集优化
- 预加载数据到内存 (`preload=True`)
- 使用多工作进程 (`num_workers`)
- 启用内存固定 (`pin_memory=True`)
- 选择合适的数据集类型（标准vs可迭代）

## 测试结果示例

```
📊 视频数据集处理性能比较结果
================================================================================

🎯 PyTorch Dataset:
   总处理时间: 15.23 秒
   处理样本数: 20
   处理速度: 1.31 样本/秒
   内存使用: 245.67 MB
   峰值内存: 512.34 MB

🎯 Ray Dataset:
   总处理时间: 12.45 秒
   处理样本数: 20
   处理速度: 1.61 样本/秒
   内存使用: 189.23 MB
   峰值内存: 423.12 MB

🎯 HuggingFace Dataset:
   总处理时间: 18.67 秒
   处理样本数: 20
   处理速度: 1.07 样本/秒
   内存使用: 156.78 MB
   峰值内存: 389.45 MB

🏆 最快处理: Ray Dataset (1.61 样本/秒)
💾 最省内存: HuggingFace Dataset (156.78 MB)
```

## 自定义配置

### 修改视频处理参数

```python
# 修改提取帧数
num_frames = 32  # 默认16

# 修改目标尺寸
target_size = (256, 256)  # 默认(224, 224)

# 修改批处理大小
batch_size = 8  # 默认4
```

### 修改数据路径

```python
# 在脚本中修改VIDEO_DATA_PATH变量
VIDEO_DATA_PATH = "/path/to/your/video/data"
```

## 故障排除

### 常见问题

1. **内存不足错误**
   - 减少批处理大小
   - 关闭预加载功能
   - 减少工作进程数

2. **视频读取失败**
   - 检查视频文件是否损坏
   - 确认OpenCV支持的视频格式
   - 检查文件路径权限

3. **Ray初始化错误**
   - 确保Ray版本兼容
   - 检查系统资源是否充足
   - 尝试重启Ray集群

### 性能调优建议

1. **对于小数据集** (< 100个视频)
   - 使用PyTorch预加载模式
   - 启用多工作进程

2. **对于中等数据集** (100-1000个视频)
   - 使用Ray数据集
   - 适当调整批处理大小

3. **对于大数据集** (> 1000个视频)
   - 使用HuggingFace缓存功能
   - 考虑分布式处理

## 扩展功能

### 添加新的数据变换

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 支持其他视频格式

```python
# 在find_all_mp4_files函数中添加其他格式
supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
```

## 贡献指南

欢迎提交问题报告和功能请求！如果您想贡献代码，请：

1. Fork这个项目
2. 创建功能分支
3. 提交您的更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。 