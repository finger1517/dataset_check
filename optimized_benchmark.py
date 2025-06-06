#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的视频数据集基准测试
解决并发效率和HuggingFace性能问题
"""

import os
import time
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
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import tempfile
import pickle

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

def extract_frames_from_video_optimized(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """优化的视频帧提取函数"""
    cap = cv2.VideoCapture(video_path)
    
    # 优化：设置缓冲区大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # 计算均匀采样的帧索引
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    # 优化：预分配内存
    frame_array = np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 优化：直接写入预分配的数组
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_array[i] = frame_rgb
        # 如果读取失败，保持零值（已预分配）
    
    cap.release()
    return frame_array

# ==================== PyTorch Dataset 优化实现 ====================
class OptimizedPyTorchVideoDataset(Dataset):
    """优化的PyTorch视频数据集"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224), 
                 use_cache: bool = False):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        self.use_cache = use_cache
        self.cache = {}
        
        # 优化的变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # 缓存机制
        if self.use_cache and video_path in self.cache:
            frames = self.cache[video_path]
        else:
            frames = extract_frames_from_video_optimized(video_path, self.num_frames, self.target_size)
            if self.use_cache:
                self.cache[video_path] = frames
        
        # 优化：批量转换
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return {
            'frames': frames_tensor,
            'video_path': video_path,
            'video_name': os.path.basename(video_path)
        }

# ==================== Ray Dataset 优化实现 ====================
def process_video_ray_optimized(batch: Dict) -> Dict:
    """优化的Ray数据处理函数"""
    video_paths = batch['video_path']
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=min(4, len(video_paths))) as executor:
        futures = {executor.submit(extract_frames_from_video_optimized, path): path 
                  for path in video_paths}
        
        processed_batch = {
            'frames': [],
            'video_path': [],
            'video_name': []
        }
        
        for future in futures:
            video_path = futures[future]
            try:
                frames = future.result()
                
                # 优化：向量化标准化
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                processed_batch['frames'].append(frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")
                empty_frames = np.zeros((16, 224, 224, 3), dtype=np.float32)
                processed_batch['frames'].append(empty_frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
    
    return processed_batch

# ==================== HuggingFace Dataset 优化实现 ====================
def process_video_hf_lightweight(example):
    """轻量级HuggingFace数据处理函数"""
    video_path = example['video_path']
    
    try:
        # 只返回路径，延迟加载
        return {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'processed': False  # 标记未处理
        }
    except Exception as e:
        return {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'processed': False,
            'error': str(e)
        }

def create_hf_dataset_optimized(video_paths: List[str], use_lazy_loading: bool = True) -> HFDataset:
    """创建优化的HuggingFace数据集"""
    data = [{'video_path': path} for path in video_paths]
    dataset = HFDataset.from_list(data)
    
    if use_lazy_loading:
        # 使用延迟加载，只处理元数据
        return dataset.map(
            process_video_hf_lightweight, 
            num_proc=1,  # 减少进程数
            desc="准备视频元数据"
        )
    else:
        # 传统方式，完整处理
        return dataset.map(
            lambda x: {
                'frames': extract_frames_from_video_optimized(x['video_path']),
                'video_path': x['video_path'],
                'video_name': os.path.basename(x['video_path'])
            }, 
            num_proc=min(4, cpu_count()),
            desc="处理视频数据"
        )

# ==================== 性能测试函数 ====================
def measure_memory_usage():
    """测量内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_pytorch_optimized(video_paths: List[str], batch_size: int = 8, num_workers: int = 4, 
                               use_cache: bool = False) -> Dict:
    """测试优化的PyTorch数据集性能"""
    print(f"🔥 测试优化PyTorch数据集 (workers={num_workers}, batch_size={batch_size}, cache={use_cache})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = OptimizedPyTorchVideoDataset(video_paths, use_cache=use_cache)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    processed_samples = 0
    batch_count = 0
    
    for batch in dataloader:
        batch_count += 1
        processed_samples += len(batch['frames'])
        # 模拟处理
        _ = batch['frames'].mean()
        
        if batch_count % 10 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'PyTorch优化版 ({num_workers}workers, cache={use_cache})',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'num_workers': num_workers,
        'batch_size': batch_size
    }

def benchmark_ray_optimized(video_paths: List[str], batch_size: int = 8, num_workers: int = 4) -> Dict:
    """测试优化的Ray数据集性能"""
    print(f"⚡ 测试优化Ray数据集 (workers={num_workers}, batch_size={batch_size})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    data = [{'video_path': path} for path in video_paths]
    ds = ray.data.from_items(data)
    dataset = ds.map_batches(
        process_video_ray_optimized, 
        batch_size=batch_size,
        num_cpus=2,
        concurrency=num_workers
    )
    
    processed_samples = 0
    batch_count = 0
    
    for batch in dataset.iter_batches(batch_size=batch_size):
        batch_count += 1
        processed_samples += len(batch['frames'])
        _ = np.mean([np.mean(frames) for frames in batch['frames']])
        
        if batch_count % 10 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'Ray优化版 ({num_workers}workers)',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'num_workers': num_workers,
        'batch_size': batch_size
    }

def benchmark_hf_optimized(video_paths: List[str], batch_size: int = 8, use_lazy_loading: bool = True) -> Dict:
    """测试优化的HuggingFace数据集性能"""
    mode = "延迟加载" if use_lazy_loading else "完整处理"
    print(f"🤗 测试优化HuggingFace数据集 ({mode}, batch_size={batch_size})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = create_hf_dataset_optimized(video_paths, use_lazy_loading=use_lazy_loading)
    
    processed_samples = 0
    batch_count = 0
    
    if use_lazy_loading:
        # 延迟加载模式：在迭代时才处理视频
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
            batch_paths = dataset[i:i+batch_size]['video_path']
            
            # 实时处理视频
            if isinstance(batch_paths, list):
                for path in batch_paths:
                    frames = extract_frames_from_video_optimized(path)
                    _ = np.mean(frames)
                    processed_samples += 1
            else:
                frames = extract_frames_from_video_optimized(batch_paths)
                _ = np.mean(frames)
                processed_samples += 1
            
            if batch_count % 10 == 0:
                print(f"   已处理 {processed_samples} 个样本...")
    else:
        # 传统模式
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
            batch = dataset[i:i+batch_size]
            
            if isinstance(batch['frames'], list):
                processed_samples += len(batch['frames'])
                _ = np.mean([np.mean(frames) for frames in batch['frames']])
            else:
                processed_samples += 1
                _ = np.mean(batch['frames'])
            
            if batch_count % 10 == 0:
                print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'HuggingFace优化版 ({mode})',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'batch_size': batch_size,
        'lazy_loading': use_lazy_loading
    }

def run_optimized_comparison(video_paths: List[str]) -> List[Dict]:
    """运行优化的性能比较"""
    results = []
    
    print(f"🚀 开始优化性能测试 (CPU核心数: {cpu_count()})")
    
    # 测试PyTorch不同配置
    for num_workers in [2, 4, 8]:
        if num_workers <= cpu_count():
            # 无缓存版本
            result = benchmark_pytorch_optimized(video_paths, batch_size=8, num_workers=num_workers, use_cache=False)
            results.append(result)
            
            # 缓存版本（仅测试一次）
            if num_workers == 4:
                result_cached = benchmark_pytorch_optimized(video_paths, batch_size=8, num_workers=num_workers, use_cache=True)
                results.append(result_cached)
    
    # 测试Ray不同配置
    for num_workers in [2, 4, 8]:
        if num_workers <= cpu_count():
            result = benchmark_ray_optimized(video_paths, batch_size=8, num_workers=num_workers)
            results.append(result)
    
    # 测试HuggingFace优化版本
    # 延迟加载版本
    result_lazy = benchmark_hf_optimized(video_paths, batch_size=8, use_lazy_loading=True)
    results.append(result_lazy)
    
    # 完整处理版本（仅用少量数据测试）
    small_paths = video_paths[:20]  # 只用20个文件测试
    result_full = benchmark_hf_optimized(small_paths, batch_size=8, use_lazy_loading=False)
    results.append(result_full)
    
    return results

def print_optimized_results(results: List[Dict]):
    """打印优化测试结果"""
    print("\n" + "="*100)
    print("📊 优化后的视频数据集处理性能比较结果")
    print("="*100)
    
    for result in results:
        print(f"\n🎯 {result['name']}:")
        print(f"   总处理时间: {result['total_time']:.2f} 秒")
        print(f"   处理样本数: {result['samples_processed']}")
        print(f"   处理速度: {result['samples_per_second']:.2f} 样本/秒")
        print(f"   内存使用: {result['memory_usage_mb']:.2f} MB")
        print(f"   峰值内存: {result['peak_memory_mb']:.2f} MB")
    
    # 找出最佳配置
    fastest = max(results, key=lambda x: x['samples_per_second'])
    most_efficient_memory = min(results, key=lambda x: x['memory_usage_mb'])
    
    print(f"\n🏆 最快配置: {fastest['name']}")
    print(f"   处理速度: {fastest['samples_per_second']:.2f} 样本/秒")
    
    print(f"\n💾 最省内存配置: {most_efficient_memory['name']}")
    print(f"   内存使用: {most_efficient_memory['memory_usage_mb']:.2f} MB")

def main():
    """主函数"""
    print("🎬 开始优化的视频数据集处理效能比较测试...")
    print(f"💻 系统信息: {cpu_count()} CPU核心")
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=cpu_count(),
            ignore_reinit_error=True,
            include_dashboard=False
        )
    
    # 查找所有视频文件
    print("📁 正在搜索视频文件...")
    video_paths = find_all_mp4_files(VIDEO_DATA_PATH)
    
    if not video_paths:
        print("❌ 未找到任何mp4文件！")
        return
    
    # 使用适量的文件进行测试
    test_video_paths = video_paths[:min(100, len(video_paths))]
    print(f"📹 找到 {len(video_paths)} 个视频文件，将测试前 {len(test_video_paths)} 个")
    
    try:
        # 运行优化比较测试
        results = run_optimized_comparison(test_video_paths)
        
        # 打印结果
        print_optimized_results(results)
        
        # 保存结果到CSV
        df = pd.DataFrame(results)
        df.to_csv('optimized_benchmark_results.csv', index=False)
        print(f"\n💾 详细结果已保存到 optimized_benchmark_results.csv")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理Ray资源
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main() 