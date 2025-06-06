#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频数据集处理效能比较脚本
比较Ray、HuggingFace和PyTorch数据集在处理视频数据时的性能
"""

import os
import time
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import ray
from ray.data import Dataset as RayDataset
from datasets import Dataset as HFDataset
import pandas as pd
from typing import List, Tuple, Dict
import psutil
import gc
from pathlib import Path

# 设置视频数据路径
VIDEO_DATA_PATH = "../video_data"

def find_all_mp4_files(base_path: str) -> List[str]:
    """递归查找所有mp4文件"""
    mp4_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def extract_frames_from_video(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """从视频中均匀提取指定数量的帧并调整大小"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # 计算均匀采样的帧索引
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 调整大小并转换颜色空间
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # 如果读取失败，添加零帧
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames)

# ==================== PyTorch Dataset 实现 ====================
class PyTorchVideoDataset(Dataset):
    """PyTorch标准数据集实现"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = extract_frames_from_video(video_path, self.num_frames, self.target_size)
        
        # 转换为tensor并应用标准化
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        
        return {
            'frames': frames_tensor,
            'video_path': video_path,
            'video_name': os.path.basename(video_path)
        }

# ==================== Ray Dataset 实现 ====================
def process_video_ray(batch: Dict) -> Dict:
    """Ray数据处理函数"""
    video_paths = batch['video_path']
    processed_batch = {
        'frames': [],
        'video_path': [],
        'video_name': []
    }
    
    for video_path in video_paths:
        frames = extract_frames_from_video(video_path, 16, (224, 224))
        # 标准化处理
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        processed_batch['frames'].append(frames)
        processed_batch['video_path'].append(video_path)
        processed_batch['video_name'].append(os.path.basename(video_path))
    
    return processed_batch

def create_ray_dataset(video_paths: List[str]) -> RayDataset:
    """创建Ray数据集"""
    data = [{'video_path': path} for path in video_paths]
    ds = ray.data.from_items(data)
    return ds.map_batches(process_video_ray, batch_size=4)

# ==================== HuggingFace Dataset 实现 ====================
def process_video_hf(example):
    """HuggingFace数据处理函数"""
    video_path = example['video_path']
    frames = extract_frames_from_video(video_path, 16, (224, 224))
    
    # 标准化处理
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    return {
        'frames': frames,
        'video_path': video_path,
        'video_name': os.path.basename(video_path)
    }

def create_hf_dataset(video_paths: List[str]) -> HFDataset:
    """创建HuggingFace数据集"""
    data = [{'video_path': path} for path in video_paths]
    dataset = HFDataset.from_list(data)
    return dataset.map(process_video_hf, num_proc=4)

# ==================== 性能测试函数 ====================
def measure_memory_usage():
    """测量内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_pytorch_dataset(video_paths: List[str], batch_size: int = 4) -> Dict:
    """测试PyTorch数据集性能"""
    print("🔥 测试PyTorch数据集...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = PyTorchVideoDataset(video_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    processed_samples = 0
    for batch in dataloader:
        processed_samples += len(batch['frames'])
        # 模拟一些处理
        _ = batch['frames'].mean()
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': 'PyTorch Dataset',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory
    }

def benchmark_ray_dataset(video_paths: List[str], batch_size: int = 4) -> Dict:
    """测试Ray数据集性能"""
    print("⚡ 测试Ray数据集...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = create_ray_dataset(video_paths)
    
    processed_samples = 0
    for batch in dataset.iter_batches(batch_size=batch_size):
        processed_samples += len(batch['frames'])
        # 模拟一些处理
        _ = np.mean([np.mean(frames) for frames in batch['frames']])
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': 'Ray Dataset',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory
    }

def benchmark_hf_dataset(video_paths: List[str], batch_size: int = 4) -> Dict:
    """测试HuggingFace数据集性能"""
    print("🤗 测试HuggingFace数据集...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = create_hf_dataset(video_paths)
    
    processed_samples = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        if isinstance(batch['frames'], list):
            processed_samples += len(batch['frames'])
            # 模拟一些处理
            _ = np.mean([np.mean(frames) for frames in batch['frames']])
        else:
            processed_samples += 1
            _ = np.mean(batch['frames'])
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': 'HuggingFace Dataset',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory
    }

def print_results(results: List[Dict]):
    """打印测试结果"""
    print("\n" + "="*80)
    print("📊 视频数据集处理性能比较结果")
    print("="*80)
    
    for result in results:
        print(f"\n🎯 {result['name']}:")
        print(f"   总处理时间: {result['total_time']:.2f} 秒")
        print(f"   处理样本数: {result['samples_processed']}")
        print(f"   处理速度: {result['samples_per_second']:.2f} 样本/秒")
        print(f"   内存使用: {result['memory_usage_mb']:.2f} MB")
        print(f"   峰值内存: {result['peak_memory_mb']:.2f} MB")
    
    # 找出最快的方法
    fastest = min(results, key=lambda x: x['total_time'])
    most_efficient_memory = min(results, key=lambda x: x['memory_usage_mb'])
    
    print(f"\n🏆 最快处理: {fastest['name']} ({fastest['samples_per_second']:.2f} 样本/秒)")
    print(f"💾 最省内存: {most_efficient_memory['name']} ({most_efficient_memory['memory_usage_mb']:.2f} MB)")

def main():
    """主函数"""
    print("🎬 开始视频数据集处理效能比较测试...")
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 查找所有视频文件
    print("📁 正在搜索视频文件...")
    video_paths = find_all_mp4_files(VIDEO_DATA_PATH)
    
    if not video_paths:
        print("❌ 未找到任何mp4文件！")
        return
    
    # 限制测试文件数量以便快速测试
    test_video_paths = video_paths[:min(200, len(video_paths))]
    print(f"📹 找到 {len(video_paths)} 个视频文件，将测试前 {len(test_video_paths)} 个")
    
    results = []
    batch_size = 4
    
    try:
        # 测试PyTorch数据集
        gc.collect()
        pytorch_result = benchmark_pytorch_dataset(test_video_paths, batch_size)
        results.append(pytorch_result)
        
        # 测试Ray数据集
        gc.collect()
        ray_result = benchmark_ray_dataset(test_video_paths, batch_size)
        results.append(ray_result)
        
        # 测试HuggingFace数据集
        gc.collect()
        hf_result = benchmark_hf_dataset(test_video_paths, batch_size)
        results.append(hf_result)
        
        # 打印结果
        print_results(results)
        
        # 保存结果到CSV
        df = pd.DataFrame(results)
        df.to_csv('video_dataset_benchmark_results.csv', index=False)
        print(f"\n💾 结果已保存到 video_dataset_benchmark_results.csv")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
    finally:
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main() 