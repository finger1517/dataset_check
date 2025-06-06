#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ray数据集实现
使用Ray Data进行分布式视频数据处理
"""

import os
import cv2
import numpy as np
import ray
from ray.data import Dataset as RayDataset
from typing import List, Dict, Tuple
import time

class RayVideoDataset:
    """Ray视频数据集类"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化Ray视频数据集
        
        Args:
            video_paths: 视频文件路径列表
            num_frames: 每个视频提取的帧数
            target_size: 目标图像尺寸
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        
        # 初始化Ray（如果尚未初始化）
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
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
    
    def _process_video_batch(self, batch: Dict) -> Dict:
        """处理视频批次的函数"""
        video_paths = batch['video_path']
        processed_batch = {
            'frames': [],
            'video_path': [],
            'video_name': [],
            'frame_count': []
        }
        
        for video_path in video_paths:
            try:
                frames = self._extract_frames_from_video(video_path)
                
                # 标准化处理
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                processed_batch['frames'].append(frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
                processed_batch['frame_count'].append(len(frames))
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")
                # 添加空数据以保持批次大小一致
                empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                processed_batch['frames'].append(empty_frames)
                processed_batch['video_path'].append(video_path)
                processed_batch['video_name'].append(os.path.basename(video_path))
                processed_batch['frame_count'].append(0)
        
        return processed_batch
    
    def create_dataset(self, batch_size: int = 4) -> RayDataset:
        """创建Ray数据集"""
        # 创建初始数据
        data = [{'video_path': path} for path in self.video_paths]
        
        # 创建Ray数据集
        ds = ray.data.from_items(data)
        
        # 应用批处理转换
        processed_ds = ds.map_batches(
            self._process_video_batch,
            batch_size=batch_size,
            num_cpus=1  # 每个任务使用1个CPU
        )
        
        return processed_ds
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'total_videos': len(self.video_paths),
            'frames_per_video': self.num_frames,
            'target_size': self.target_size,
            'total_frames': len(self.video_paths) * self.num_frames
        }

def benchmark_ray_dataset_advanced(video_paths: List[str], batch_size: int = 4, num_workers: int = 2) -> Dict:
    """高级Ray数据集性能测试"""
    print(f"⚡ 开始Ray数据集高级性能测试 (batch_size={batch_size}, workers={num_workers})...")
    
    start_time = time.time()
    
    # 创建Ray视频数据集
    ray_dataset = RayVideoDataset(video_paths)
    dataset = ray_dataset.create_dataset(batch_size=batch_size)
    # dataset = dataset.cache()
    
    # 配置并行处理
    dataset = dataset.repartition(num_workers)
    
    processed_samples = 0
    batch_count = 0
    
    # 遍历数据集
    for batch in dataset.iter_batches(batch_size=batch_size):
        batch_count += 1
        current_batch_size = len(batch['frames'])
        processed_samples += current_batch_size
        
        # 模拟一些计算
        for frames in batch['frames']:
            _ = np.mean(frames)
        
        if batch_count % 5 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取数据集统计信息
    stats = ray_dataset.get_stats()
    
    return {
        'name': f'Ray Dataset (Advanced)',
        'total_time': total_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / total_time if total_time > 0 else 0,
        'batch_count': batch_count,
        'avg_batch_size': processed_samples / batch_count if batch_count > 0 else 0,
        'dataset_stats': stats
    }

if __name__ == "__main__":
    # 测试Ray数据集实现
    import glob
    
    # 查找测试视频文件
    video_paths = []
    for root, dirs, files in os.walk("../video_data"):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    if video_paths:
        test_paths = video_paths[:400]  # 测试前5个视频
        print(f"找到 {len(video_paths)} 个视频文件，测试前 {len(test_paths)} 个")
        
        # 运行测试
        result = benchmark_ray_dataset_advanced(test_paths, batch_size=4, num_workers=50)
        
        print("\n测试结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("未找到视频文件进行测试")
