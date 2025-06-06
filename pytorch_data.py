#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch数据集实现
使用PyTorch Dataset和DataLoader进行视频数据处理
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from typing import List, Dict, Tuple, Iterator
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class PyTorchVideoDataset(Dataset):
    """PyTorch标准视频数据集类"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224), 
                 transform=None, preload: bool = False):
        """
        初始化PyTorch视频数据集
        
        Args:
            video_paths: 视频文件路径列表
            num_frames: 每个视频提取的帧数
            target_size: 目标图像尺寸
            transform: 数据变换
            preload: 是否预加载所有数据到内存
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        self.preload = preload
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 预加载数据
        if self.preload:
            print("预加载视频数据到内存...")
            self.preloaded_data = self._preload_all_videos()
            print(f"预加载完成，共 {len(self.preloaded_data)} 个视频")
    
    def _extract_frames_from_video(self, video_path: str) -> np.ndarray:
        """从视频中均匀提取帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # 计算均匀采样的帧索引
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 调整大小并转换颜色空间
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果读取失败，添加零帧
                frames.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def _preload_all_videos(self) -> List[Dict]:
        """预加载所有视频数据"""
        preloaded_data = []
        
        with ProcessPoolExecutor(max_workers=40) as executor:
            futures = []
            for video_path in self.video_paths:
                future = executor.submit(self._extract_frames_from_video, video_path)
                futures.append((future, video_path))
            
            for future, video_path in futures:
                try:
                    frames = future.result()
                    preloaded_data.append({
                        'frames': frames,
                        'video_path': video_path,
                        'video_name': os.path.basename(video_path)
                    })
                except Exception as e:
                    print(f"预加载视频 {video_path} 时出错: {str(e)}")
                    empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
                    preloaded_data.append({
                        'frames': empty_frames,
                        'video_path': video_path,
                        'video_name': os.path.basename(video_path)
                    })
        
        return preloaded_data
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        if self.preload:
            # 使用预加载的数据
            data = self.preloaded_data[idx]
            frames = data['frames']
            video_path = data['video_path']
            video_name = data['video_name']
        else:
            # 实时加载
            video_path = self.video_paths[idx]
            frames = self._extract_frames_from_video(video_path)
            video_name = os.path.basename(video_path)
        
        # 转换为tensor并应用变换
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        
        return {
            'frames': frames_tensor,
            'video_path': video_path,
            'video_name': video_name,
            'video_idx': idx
        }

class PyTorchIterableVideoDataset(IterableDataset):
    """PyTorch可迭代视频数据集类（适合大规模数据）"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224), 
                 transform=None, shuffle: bool = False):
        """
        初始化PyTorch可迭代视频数据集
        
        Args:
            video_paths: 视频文件路径列表
            num_frames: 每个视频提取的帧数
            target_size: 目标图像尺寸
            transform: 数据变换
            shuffle: 是否打乱数据
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        self.shuffle = shuffle
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _extract_frames_from_video(self, video_path: str) -> np.ndarray:
        """从视频中均匀提取帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # 计算均匀采样的帧索引
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 调整大小并转换颜色空间
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果读取失败，添加零帧
                frames.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def __iter__(self) -> Iterator[Dict]:
        video_paths = self.video_paths.copy()
        
        if self.shuffle:
            np.random.shuffle(video_paths)
        
        for idx, video_path in enumerate(video_paths):
            try:
                frames = self._extract_frames_from_video(video_path)
                
                # 转换为tensor并应用变换
                frames_tensor = torch.stack([self.transform(frame) for frame in frames])
                
                yield {
                    'frames': frames_tensor,
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'video_idx': idx
                }
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")
                # 返回空数据
                empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
                frames_tensor = torch.stack([self.transform(frame) for frame in empty_frames])
                
                yield {
                    'frames': frames_tensor,
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'video_idx': idx
                }

def benchmark_pytorch_dataset_advanced(video_paths: List[str], batch_size: int = 4, num_workers: int = 2, 
                                     preload: bool = False, use_iterable: bool = False) -> Dict:
    """高级PyTorch数据集性能测试"""
    dataset_type = "Iterable" if use_iterable else ("Preloaded" if preload else "Standard")
    print(f"🔥 开始PyTorch数据集高级性能测试 ({dataset_type}, batch_size={batch_size}, workers={num_workers})...")
    
    start_time = time.time()
    
    # 创建数据集
    if use_iterable:
        dataset = PyTorchIterableVideoDataset(video_paths, shuffle=True)
    else:
        dataset = PyTorchVideoDataset(video_paths, preload=preload)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not use_iterable,  # 可迭代数据集内部处理shuffle
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    processed_samples = 0
    batch_count = 0
    
    # 遍历数据加载器
    for batch in dataloader:
        batch_count += 1
        current_batch_size = len(batch['frames'])
        processed_samples += current_batch_size
        
        # 模拟一些计算
        _ = batch['frames'].mean()
        print(type(batch['frames']))
        
        # # 如果有GPU，测试GPU传输
        # if torch.cuda.is_available():
        #     gpu_frames = batch['frames'].cuda()
        #     _ = gpu_frames.mean()
        #     del gpu_frames
        
        if batch_count % 5 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'name': f'PyTorch Dataset ({dataset_type})',
        'total_time': total_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / total_time if total_time > 0 else 0,
        'batch_count': batch_count,
        'avg_batch_size': processed_samples / batch_count if batch_count > 0 else 0,
        'num_workers': num_workers,
        'preload': preload,
        'use_iterable': use_iterable,
        'gpu_available': torch.cuda.is_available()
    }

def compare_pytorch_variants(video_paths: List[str], batch_size: int = 4) -> List[Dict]:
    """比较不同PyTorch数据集变体的性能"""
    results = []
    
    # 测试标准数据集
    result1 = benchmark_pytorch_dataset_advanced(
        video_paths, batch_size=batch_size, num_workers=0, preload=False, use_iterable=False
    )
    results.append(result1)
    
    # 测试多进程数据集
    result2 = benchmark_pytorch_dataset_advanced(
        video_paths, batch_size=batch_size, num_workers=2, preload=False, use_iterable=False
    )
    results.append(result2)
    
    # 测试预加载数据集
    result3 = benchmark_pytorch_dataset_advanced(
        video_paths, batch_size=batch_size, num_workers=0, preload=True, use_iterable=False
    )
    results.append(result3)
    
    # 测试可迭代数据集
    result4 = benchmark_pytorch_dataset_advanced(
        video_paths, batch_size=batch_size, num_workers=2, preload=False, use_iterable=True
    )
    results.append(result4)
    
    return results

if __name__ == "__main__":
    # 测试PyTorch数据集实现
    import glob
    
    # 查找测试视频文件
    video_paths = []
    for root, dirs, files in os.walk("../video_data"):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    if video_paths:
        test_paths = video_paths[:50]  # 测试前50个视频
        print(f"找到 {len(video_paths)} 个视频文件，测试前 {len(test_paths)} 个")
        
        # 比较不同变体
        # results = compare_pytorch_variants(test_paths, batch_size=4)
        result = benchmark_pytorch_dataset_advanced(test_paths, batch_size=4, num_workers=40, preload=False, use_iterable=False)
        
        # print("\n" + "="*60)
        # print("PyTorch数据集变体性能比较结果")
        # print("="*60)
        
        print(f"\n🎯 {result['name']}:")
        print(f"   总处理时间: {result['total_time']:.2f} 秒")
        print(f"   处理样本数: {result['samples_processed']}")
        print(f"   处理速度: {result['samples_per_second']:.2f} 样本/秒")
        print(f"   工作进程数: {result['num_workers']}")
        print(f"   预加载: {result['preload']}")
        print(f"   可迭代: {result['use_iterable']}")
        print(f"   GPU可用: {result['gpu_available']}")
        
        # 找出最快的变体
        # fastest = min(results, key=lambda x: x['total_time'])
        # print(f"\n🏆 最快变体: {fastest['name']} ({fastest['samples_per_second']:.2f} 样本/秒)")
        
    else:
        print("未找到视频文件进行测试") 