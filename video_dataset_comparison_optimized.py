#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频数据集处理效能比较脚本 - 并发优化版本
充分利用Ray、HuggingFace和PyTorch数据集的并发能力
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
from concurrent.futures import ThreadPoolExecutor
import threading

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

# ==================== PyTorch Dataset 优化实现 ====================
class OptimizedPyTorchVideoDataset(Dataset):
    """优化的PyTorch视频数据集实现"""
    
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

def collate_fn(batch):
    """自定义批处理函数，优化内存使用"""
    frames = torch.stack([item['frames'] for item in batch])
    video_paths = [item['video_path'] for item in batch]
    video_names = [item['video_name'] for item in batch]
    
    return {
        'frames': frames,
        'video_paths': video_paths,
        'video_names': video_names
    }

# ==================== Ray Dataset 优化实现 ====================
def process_video_ray_optimized(batch: Dict) -> Dict:
    """Ray数据处理函数 - 优化版本，使用线程池"""
    video_paths = batch['video_path']
    processed_batch = {
        'frames': [],
        'video_path': [],
        'video_name': []
    }
    
    # 使用线程池并行处理视频
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for video_path in video_paths:
            future = executor.submit(extract_frames_from_video, video_path, 16, (224, 224))
            futures.append((future, video_path))
        
        for future, video_path in futures:
            try:
                frames = future.result()
                # 标准化处理
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                processed_batch['frames'].append(frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")
                # 添加空数据
                empty_frames = np.zeros((16, 224, 224, 3), dtype=np.float32)
                processed_batch['frames'].append(empty_frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
    
    return processed_batch

def create_ray_dataset_optimized(video_paths: List[str], num_workers: int = None) -> RayDataset:
    """创建优化的Ray数据集"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    data = [{'video_path': path} for path in video_paths]
    ds = ray.data.from_items(data)
    
    # 使用更大的批处理大小和更多并行度
    return ds.map_batches(
        process_video_ray_optimized, 
        batch_size=8,
        num_cpus=2,  # 每个任务使用2个CPU
        concurrency=num_workers
    )

# ==================== HuggingFace Dataset 优化实现 ====================
def process_video_hf_optimized(example):
    """HuggingFace数据处理函数 - 优化版本"""
    video_path = example['video_path']
    
    try:
        frames = extract_frames_from_video(video_path, 16, (224, 224))
        
        # 标准化处理
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        return {
            'frames': frames,
            'video_path': video_path,
            'video_name': os.path.basename(video_path)
        }
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {str(e)}")
        empty_frames = np.zeros((16, 224, 224, 3), dtype=np.float32)
        return {
            'frames': empty_frames,
            'video_path': video_path,
            'video_name': os.path.basename(video_path)
        }

def create_hf_dataset_optimized(video_paths: List[str], num_proc: int = None) -> HFDataset:
    """创建优化的HuggingFace数据集"""
    if num_proc is None:
        num_proc = min(cpu_count(), 8)
    
    data = [{'video_path': path} for path in video_paths]
    dataset = HFDataset.from_list(data)
    
    # 使用更多进程和批处理
    return dataset.map(
        process_video_hf_optimized, 
        num_proc=num_proc,
        batch_size=1000,  # 更大的批处理大小
        desc="处理视频数据"
    )

# ==================== 性能测试函数 ====================
def measure_memory_usage():
    """测量内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_pytorch_dataset_optimized(video_paths: List[str], batch_size: int = 8, num_workers: int = None) -> Dict:
    """测试优化的PyTorch数据集性能"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    print(f"🔥 测试优化PyTorch数据集 (workers={num_workers}, batch_size={batch_size})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = OptimizedPyTorchVideoDataset(video_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2
    )
    
    processed_samples = 0
    batch_count = 0
    
    for batch in dataloader:
        batch_count += 1
        processed_samples += len(batch['frames'])
        # 模拟一些处理
        _ = batch['frames'].mean()
        
        if batch_count % 10 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'PyTorch Dataset (优化版, {num_workers}workers)',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'num_workers': num_workers,
        'batch_size': batch_size
    }

def benchmark_ray_dataset_optimized(video_paths: List[str], batch_size: int = 8, num_workers: int = None) -> Dict:
    """测试优化的Ray数据集性能"""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    print(f"⚡ 测试优化Ray数据集 (workers={num_workers}, batch_size={batch_size})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = create_ray_dataset_optimized(video_paths, num_workers)
    
    processed_samples = 0
    batch_count = 0
    
    for batch in dataset.iter_batches(batch_size=batch_size):
        batch_count += 1
        processed_samples += len(batch['frames'])
        # 模拟一些处理
        _ = np.mean([np.mean(frames) for frames in batch['frames']])
        
        if batch_count % 10 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'Ray Dataset (优化版, {num_workers}workers)',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'num_workers': num_workers,
        'batch_size': batch_size
    }

def benchmark_hf_dataset_optimized(video_paths: List[str], batch_size: int = 8, num_proc: int = None) -> Dict:
    """测试优化的HuggingFace数据集性能"""
    if num_proc is None:
        num_proc = min(cpu_count(), 8)
    
    print(f"🤗 测试优化HuggingFace数据集 (processes={num_proc}, batch_size={batch_size})...")
    
    start_memory = measure_memory_usage()
    start_time = time.time()
    
    dataset = create_hf_dataset_optimized(video_paths, num_proc)
    
    processed_samples = 0
    batch_count = 0
    
    # 使用更高效的批处理迭代
    for i in range(0, len(dataset), batch_size):
        batch_count += 1
        batch = dataset[i:i+batch_size]
        
        if isinstance(batch['frames'], list):
            processed_samples += len(batch['frames'])
            # 模拟一些处理
            _ = np.mean([np.mean(frames) for frames in batch['frames']])
        else:
            processed_samples += 1
            _ = np.mean(batch['frames'])
        
        if batch_count % 10 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    end_memory = measure_memory_usage()
    
    return {
        'name': f'HuggingFace Dataset (优化版, {num_proc}processes)',
        'total_time': end_time - start_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / (end_time - start_time),
        'memory_usage_mb': end_memory - start_memory,
        'peak_memory_mb': end_memory,
        'num_workers': num_proc,
        'batch_size': batch_size
    }

def run_concurrent_comparison(video_paths: List[str]) -> List[Dict]:
    """运行并发性能比较测试"""
    results = []
    
    # 测试不同的并发配置
    worker_configs = [2, 4, 8] if cpu_count() >= 8 else [2, 4]
    batch_sizes = [4, 8, 16]
    
    print(f"🚀 开始并发性能测试 (CPU核心数: {cpu_count()})")
    print(f"测试配置: workers={worker_configs}, batch_sizes={batch_sizes}")
    
    for num_workers in worker_configs:
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"测试配置: {num_workers} workers, batch_size={batch_size}")
            print(f"{'='*60}")
            
            # 测试PyTorch
            gc.collect()
            pytorch_result = benchmark_pytorch_dataset_optimized(
                video_paths, batch_size=batch_size, num_workers=num_workers
            )
            results.append(pytorch_result)
            
            # 测试Ray
            gc.collect()
            ray_result = benchmark_ray_dataset_optimized(
                video_paths, batch_size=batch_size, num_workers=num_workers
            )
            results.append(ray_result)
            
            # 测试HuggingFace
            gc.collect()
            hf_result = benchmark_hf_dataset_optimized(
                video_paths, batch_size=batch_size, num_proc=num_workers
            )
            results.append(hf_result)
    
    return results

def print_detailed_results(results: List[Dict]):
    """打印详细的测试结果"""
    print("\n" + "="*100)
    print("📊 视频数据集处理并发性能比较结果")
    print("="*100)
    
    # 按框架分组显示结果
    frameworks = {}
    for result in results:
        framework = result['name'].split(' ')[0]
        if framework not in frameworks:
            frameworks[framework] = []
        frameworks[framework].append(result)
    
    for framework, framework_results in frameworks.items():
        print(f"\n🎯 {framework} 框架结果:")
        print("-" * 80)
        
        for result in framework_results:
            print(f"   配置: {result['num_workers']}workers, batch={result['batch_size']}")
            print(f"   处理时间: {result['total_time']:.2f}秒")
            print(f"   处理速度: {result['samples_per_second']:.2f}样本/秒")
            print(f"   内存使用: {result['memory_usage_mb']:.2f}MB")
            print()
    
    # 找出最佳配置
    fastest = max(results, key=lambda x: x['samples_per_second'])
    most_efficient_memory = min(results, key=lambda x: x['memory_usage_mb'])
    
    print(f"\n🏆 最快配置: {fastest['name']}")
    print(f"   处理速度: {fastest['samples_per_second']:.2f} 样本/秒")
    print(f"   配置: {fastest['num_workers']}workers, batch={fastest['batch_size']}")
    
    print(f"\n💾 最省内存配置: {most_efficient_memory['name']}")
    print(f"   内存使用: {most_efficient_memory['memory_usage_mb']:.2f} MB")
    print(f"   配置: {most_efficient_memory['num_workers']}workers, batch={most_efficient_memory['batch_size']}")

def main():
    """主函数"""
    print("🎬 开始视频数据集处理并发效能比较测试...")
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
    
    # 限制测试文件数量
    test_video_paths = video_paths[:min(200, len(video_paths))]
    print(f"📹 找到 {len(video_paths)} 个视频文件，将测试前 {len(test_video_paths)} 个")
    
    try:
        # 运行并发比较测试
        results = run_concurrent_comparison(test_video_paths)
        
        # 打印详细结果
        print_detailed_results(results)
        
        # 保存结果到CSV
        df = pd.DataFrame(results)
        df.to_csv('video_dataset_concurrent_benchmark_results.csv', index=False)
        print(f"\n💾 详细结果已保存到 video_dataset_concurrent_benchmark_results.csv")
        
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