#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace数据集实现
使用HuggingFace Datasets库进行视频数据处理
"""

import os
import cv2
import numpy as np
from datasets import Dataset as HFDataset, Features, Array3D, Value, IterableDataset
from typing import List, Dict, Tuple, Iterator, Any
import time
from multiprocessing import cpu_count
import polars as pl
import torch

class HuggingFaceVideoDataset:
    """HuggingFace视频数据集类"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化HuggingFace视频数据集
        
        Args:
            video_paths: 视频文件路径列表
            num_frames: 每个视频提取的帧数
            target_size: 目标图像尺寸
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
    
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
    
    def _process_video_example(self, example):
        """处理单个视频样本"""
        video_path = example['video_path']
        
        try:
            frames = self._extract_frames_from_video(video_path)
            
            # 标准化处理
            frames = frames.astype(np.float32) / 255.0
            frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            return {
                'frames': frames,
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'frame_count': len(frames),
                'video_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
            
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {str(e)}")
            # 返回空数据
            empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            return {
                'frames': empty_frames,
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'frame_count': 0,
                'video_size': 0
            }
    
    def create_dataset(self, num_proc: int = None, cache_dir: str = None) -> HFDataset:
        """创建HuggingFace数据集"""
        if num_proc is None:
            num_proc = min(4, cpu_count())
        
        # 创建初始数据
        data = [{'video_path': path} for path in self.video_paths]
        
        # 定义数据集特征
        features = Features({
            'frames': Array3D(shape=(self.num_frames, self.target_size[0], self.target_size[1], 3), dtype='float32'),
            'video_path': Value('string'),
            'video_name': Value('string'),
            'frame_count': Value('int32'),
            'video_size': Value('int64')
        })
        
        # 创建HuggingFace数据集
        dataset = HFDataset.from_list(data)
        
        # 应用处理函数
        processed_dataset = dataset.map(
            self._process_video_example,
            num_proc=num_proc,
            # batched=True,
            # batch_size=4,
            # features=features,
            # cache_file_name=f"{cache_dir}/processed_videos.arrow" if cache_dir else None,
            desc="处理视频数据"
        ).to_iterable_dataset()
        
        return processed_dataset
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'total_videos': len(self.video_paths),
            'frames_per_video': self.num_frames,
            'target_size': self.target_size,
            'total_frames': len(self.video_paths) * self.num_frames
        }

def benchmark_hf_dataset_advanced(video_paths: List[str], batch_size: int = 4, num_proc: int = 4) -> Dict:
    """高级HuggingFace数据集性能测试"""
    print(f"🤗 开始HuggingFace数据集高级性能测试 (batch_size={batch_size}, num_proc={num_proc})...")
    
    start_time = time.time()
    
    # 创建HuggingFace视频数据集
    # hf_dataset = HuggingFaceVideoDataset(video_paths)
    # dataset = hf_dataset.create_dataset(num_proc=num_proc)    
    hf_dataset = StreamingVideoDataset(video_paths)
    dataset = hf_dataset.create_streaming_dataset()
    processed_samples = 0
    batch_count = 0
    
    # HuggingFace Datasets 支持 batch 方式处理，可以通过 map 的 batched=True 参数实现批量处理
    # 这里演示如何用 batch 方式遍历数据集
    for batch in dataset.iter(batch_size=batch_size):
        batch_count += 1

        # HuggingFace 的 batch 是字典，每个 key 对应一个 list
        current_batch_size = len(batch['frames'])
        processed_samples += current_batch_size
        
        # for frames in batch['frames']:
        #     _ = np.mean(frames)

        # print(type(batch['frames']))

        # 使用polars进行批量计算
        # frames_df = pl.DataFrame({'frames': batch['frames']})
        # _ = frames_df.select(pl.col('frames').mean())

        if batch_count % 5 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    end_time = time.time()
    total_time = end_time - start_time
    
    # 获取数据集统计信息
    # stats = hf_dataset.get_stats()
    
    return {
        'name': f'HuggingFace Dataset (Advanced)',
        'total_time': total_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / total_time if total_time > 0 else 0,
        'batch_count': batch_count,
        'avg_batch_size': processed_samples / batch_count if batch_count > 0 else 0,
        # 'dataset_stats': stats,
        # 'dataset_size_mb': dataset.data.nbytes / (1024 * 1024) if hasattr(dataset.data, 'nbytes') else 0
    }

def create_hf_dataset_with_caching(video_paths: List[str], cache_dir: str = "./hf_cache") -> HFDataset:
    """创建带缓存的HuggingFace数据集"""
    os.makedirs(cache_dir, exist_ok=True)
    
    hf_dataset = HuggingFaceVideoDataset(video_paths)
    dataset = hf_dataset.create_dataset(num_proc=4, cache_dir=cache_dir)
    
    # 保存数据集到磁盘
    dataset.save_to_disk(f"{cache_dir}/video_dataset")
    print(f"数据集已保存到 {cache_dir}/video_dataset")
    
    return dataset

def load_hf_dataset_from_cache(cache_dir: str = "./hf_cache") -> HFDataset:
    """从缓存加载HuggingFace数据集"""
    dataset_path = f"{cache_dir}/video_dataset"
    if os.path.exists(dataset_path):
        dataset = HFDataset.load_from_disk(dataset_path)
        print(f"从缓存加载数据集: {dataset_path}")
        return dataset
    else:
        raise FileNotFoundError(f"缓存数据集不存在: {dataset_path}")

class OptimizedHuggingFaceVideoDataset:
    """优化的HuggingFace视频数据集，支持高效的tensor输出"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
        
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

    
    def _process_video_example_optimized(self, example):
        """优化的视频处理：使用numpy避免tensor序列化开销"""
        video_path = example['video_path']
        
        try:
            frames = self._extract_frames_from_video(video_path)
            
            # 标准化处理 - 保持numpy格式
            frames = frames.astype(np.float32) / 255.0
            frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            return {
                'frames': frames,  # numpy格式，序列化效率更高
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'frame_count': len(frames),
                'video_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
            
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {str(e)}")
            empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            return {
                'frames': empty_frames,
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'frame_count': 0,
                'video_size': 0
            }
    
    def create_torch_dataset(self, num_proc: int = None, cache_dir: str = None) -> HFDataset:
        """创建优化的torch格式数据集"""
        if num_proc is None:
            num_proc = min(4, cpu_count())
        
        data = [{'video_path': path} for path in self.video_paths]
        dataset = HFDataset.from_list(data)
        
        # 步骤1: 使用numpy处理数据（高效序列化）
        processed_dataset = dataset.map(
            self._process_video_example_optimized,
            num_proc=num_proc,
            desc="处理视频数据"
        )
        
        # 步骤2: 设置输出格式为torch（运行时转换）
        torch_dataset = processed_dataset.with_format("torch", columns=["frames"])
        
        return torch_dataset

# 使用示例
def benchmark_optimized_torch_dataset(video_paths: List[str], batch_size: int = 4, num_proc: int = 4) -> Dict:
    """测试优化的torch格式数据集"""
    print(f"🚀 测试优化torch格式HuggingFace数据集 (batch_size={batch_size}, num_proc={num_proc})...")
    
    start_time = time.time()
    
    # 创建优化的torch数据集
    hf_dataset = OptimizedHuggingFaceVideoDataset(video_paths)
    dataset = hf_dataset.create_torch_dataset(num_proc=num_proc)
    
    processed_samples = 0
    batch_count = 0
    
    for batch in dataset.iter(batch_size=batch_size):
        batch_count += 1
        current_batch_size = len(batch['frames'])
        processed_samples += current_batch_size
        
        # 验证数据类型
        print(type(batch['frames']))
        # frames = batch['frames'][0]
        # print(f"数据类型: {type(frames)}, shape: {frames.shape}")  # torch.Tensor
        
        # 可以直接使用torch操作
        # _ = torch.mean(frames)  # 无需类型转换
        
        if batch_count % 5 == 0:
            print(f"   已处理 {processed_samples} 个样本...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'name': 'HuggingFace Dataset (Optimized Torch)',
        'total_time': total_time,
        'samples_processed': processed_samples,
        'samples_per_second': processed_samples / total_time if total_time > 0 else 0,
        'batch_count': batch_count,
        'avg_batch_size': processed_samples / batch_count if batch_count > 0 else 0
    }

class StreamingVideoDataset:
    """流式视频数据集 - 避免预先处理所有数据"""
    
    def __init__(self, video_paths: List[str], num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.target_size = target_size
    
    def _extract_frames_from_video(self, video_path: str) -> np.ndarray:
        """从视频中提取帧（与之前相同的实现）"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def video_generator(self) -> Iterator[Dict[str, Any]]:
        """视频数据生成器 - 实时处理，无预存储"""
        for video_path in self.video_paths:
            try:
                # 实时处理视频
                frames = self._extract_frames_from_video(video_path)
                
                # 标准化处理
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                
                yield {
                    'frames': frames,
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'frame_count': len(frames),
                    'video_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
                }
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")
                # 返回空数据
                empty_frames = np.zeros((self.num_frames, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
                yield {
                    'frames': empty_frames,
                    'video_path': video_path,
                    'video_name': os.path.basename(video_path),
                    'frame_count': 0,
                    'video_size': 0
                }
    
    def create_streaming_dataset(self) -> IterableDataset:
        """创建流式数据集"""
        # 🔥 关键：使用生成器创建可迭代数据集
        iterable_dataset = IterableDataset.from_generator(
            self.video_generator,
            # 可选：定义特征类型（用于类型检查和优化）
            # features=Features({
            #     'frames': Array3D(shape=(self.num_frames, self.target_size[0], self.target_size[1], 3), dtype='float32'),
            #     'video_path': Value('string'),
            #     'video_name': Value('string'),
            #     'frame_count': Value('int32'),
            #     'video_size': Value('int64')
            # })
        )
        return iterable_dataset

if __name__ == "__main__":
    # 测试HuggingFace数据集实现
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
        
        # 运行测试
        result = benchmark_hf_dataset_advanced(test_paths, batch_size=4, num_proc=40)
        # result = benchmark_optimized_torch_dataset(test_paths, batch_size=4, num_proc=40)
        
        print("\n测试结果:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        # 测试缓存功能
        print("\n测试缓存功能...")
        try:
            cached_dataset = create_hf_dataset_with_caching(test_paths[:3])
            print(f"缓存数据集大小: {len(cached_dataset)}")
            
            # 尝试从缓存加载
            loaded_dataset = load_hf_dataset_from_cache()
            print(f"从缓存加载的数据集大小: {len(loaded_dataset)}")
            
        except Exception as e:
            print(f"缓存测试出错: {str(e)}")
    else:
        print("未找到视频文件进行测试") 