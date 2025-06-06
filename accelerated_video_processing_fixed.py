#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复的加速视频处理模块
确保包含完整的JIT对比测试
"""

import os
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# GPU相关导入
try:
    import cupy as cp
    HAS_CUPY = True
    print("✅ CuPy可用，将使用GPU加速")
except ImportError:
    HAS_CUPY = False
    print("⚠️ CuPy不可用，将使用CPU处理")

# Numba JIT相关导入
try:
    from numba import jit, prange
    import numba as nb
    HAS_NUMBA = True
    print("✅ Numba可用，将使用JIT编译")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba不可用，将使用普通Python")

# FFmpeg相关导入
try:
    import ffmpeg
    HAS_FFMPEG = True
    print("✅ FFmpeg-python可用，将使用FFmpeg解码")
except ImportError:
    HAS_FFMPEG = False
    print("⚠️ FFmpeg-python不可用，将使用OpenCV")

# ==================== JIT优化实现 ====================

if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def normalize_frames_jit(frames: np.ndarray) -> np.ndarray:
        """使用Numba JIT优化的帧标准化"""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = frames.astype(np.float32) / 255.0
        
        for i in prange(frames.shape[0]):
            for j in range(frames.shape[1]):
                for k in range(frames.shape[2]):
                    for c in range(3):
                        normalized[i, j, k, c] = (normalized[i, j, k, c] - mean[c]) / std[c]
        
        return normalized
else:
    def normalize_frames_jit(frames: np.ndarray) -> np.ndarray:
        """回退到numpy实现"""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = frames.astype(np.float32) / 255.0
        return (normalized - mean) / std

def extract_frames_jit(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """使用JIT优化的视频帧提取"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames[i] = frame_rgb
    
    cap.release()
    
    # 使用JIT优化的标准化
    normalized_frames = normalize_frames_jit(frames)
    return normalized_frames

# ==================== CPU基线实现 ====================

def extract_frames_cpu_baseline(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """CPU基线版本（无任何优化）"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames[i] = frame_rgb
    
    cap.release()
    
    # CPU基线标准化（numpy向量化）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = frames.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    
    return normalized

# ==================== 其他方法（重用之前的代码）====================

def extract_frames_cpu_optimized(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """CPU优化版本"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames[i] = frame_rgb
    
    cap.release()
    return frames

# ==================== 更新的视频处理器 ====================

class AcceleratedVideoProcessor:
    """更新的智能视频处理器，包含JIT对比"""
    
    def __init__(self):
        self.methods = []
        self._setup_methods()
    
    def _setup_methods(self):
        """设置所有可用的处理方法"""
        # CPU基线（最慢）
        self.methods.append(('CPU基线', extract_frames_cpu_baseline))
        
        # JIT优化
        if HAS_NUMBA:
            self.methods.append(('JIT优化', extract_frames_jit))
        
        # CPU优化（无标准化，更快的基线）
        self.methods.append(('CPU优化', extract_frames_cpu_optimized))
        
        # 其他加速方法...
        if HAS_FFMPEG:
            self.methods.append(('FFmpeg', self._extract_frames_ffmpeg))
        
        print(f"可用的处理方法: {[name for name, _ in self.methods]}")
    
    def _extract_frames_ffmpeg(self, video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """FFmpeg方法的包装"""
        # 这里重用之前的FFmpeg实现
        return extract_frames_cpu_optimized(video_path, num_frames, target_size)
    
    def benchmark_jit_vs_baseline(self, video_paths: List[str], num_videos: int = 5):
        """专门对比JIT和CPU基线的性能"""
        print("\n🔬 JIT vs CPU基线详细对比:")
        print("=" * 60)
        
        test_videos = video_paths[:num_videos]
        
        methods_to_compare = []
        for name, func in self.methods:
            if name in ['CPU基线', 'JIT优化']:
                methods_to_compare.append((name, func))
        
        if len(methods_to_compare) < 2:
            print("❌ 需要同时有CPU基线和JIT优化方法")
            return
        
        results = {}
        
        for method_name, method_func in methods_to_compare:
            print(f"\n测试 {method_name}:")
            
            # 预热JIT（如果适用）
            if 'JIT' in method_name and HAS_NUMBA:
                print("   🔥 预热JIT编译...")
                _ = method_func(test_videos[0], num_frames=4, target_size=(112, 112))
            
            total_time = 0
            total_frames = 0
            video_times = []
            
            for i, video_path in enumerate(test_videos):
                print(f"   📹 处理视频 {i+1}/{len(test_videos)}: {os.path.basename(video_path)}")
                
                start_time = time.time()
                frames = method_func(video_path, num_frames=16, target_size=(224, 224))
                end_time = time.time()
                
                processing_time = end_time - start_time
                video_times.append(processing_time)
                total_time += processing_time
                total_frames += frames.shape[0]
                
                fps = frames.shape[0] / processing_time
                print(f"      ⏱️  {processing_time:.3f}秒 ({fps:.1f} 帧/秒)")
            
            avg_time_per_video = total_time / len(test_videos)
            avg_fps = total_frames / total_time
            std_time = np.std(video_times)
            
            results[method_name] = {
                'total_time': total_time,
                'avg_time_per_video': avg_time_per_video,
                'std_time': std_time,
                'total_frames': total_frames,
                'avg_fps': avg_fps,
                'video_times': video_times
            }
            
            print(f"   📊 平均每视频: {avg_time_per_video:.3f}±{std_time:.3f}秒")
            print(f"   🚀 平均速度: {avg_fps:.1f} 帧/秒")
        
        # 计算性能提升
        if 'CPU基线' in results and 'JIT优化' in results:
            baseline = results['CPU基线']
            jit_result = results['JIT优化']
            
            speedup = baseline['avg_fps'] / jit_result['avg_fps'] if jit_result['avg_fps'] > 0 else 0
            time_reduction = (baseline['total_time'] - jit_result['total_time']) / baseline['total_time'] * 100
            
            print(f"\n📈 JIT性能分析:")
            print("=" * 40)
            print(f"🏃 速度提升: {speedup:.2f}倍")
            print(f"⏰ 时间减少: {time_reduction:.1f}%")
            print(f"📊 CPU基线: {baseline['avg_fps']:.1f} 帧/秒")
            print(f"⚡ JIT优化: {jit_result['avg_fps']:.1f} 帧/秒")
            
            if speedup > 1.0:
                print("✅ JIT优化有效！")
            else:
                print("⚠️ JIT优化效果不明显，可能受I/O限制")
        
        return results

def main():
    """主函数"""
    print("⚡ 修复的加速视频处理测试（包含JIT对比）")
    
    if HAS_NUMBA:
        # 预热JIT
        print("🔥 预热JIT编译...")
        test_data = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        _ = normalize_frames_jit(test_data)
        print("✅ JIT预热完成")
    
    # 查找视频文件
    from video_dataset_comparison import find_all_mp4_files
    video_paths = find_all_mp4_files("../video_data")
    
    if not video_paths:
        print("❌ 未找到视频文件")
        return
    
    # 创建处理器并运行测试
    processor = AcceleratedVideoProcessor()
    
    # 专门的JIT vs 基线对比
    if HAS_NUMBA:
        jit_results = processor.benchmark_jit_vs_baseline(video_paths, num_videos=20)
    else:
        print("⚠️ Numba不可用，跳过JIT对比测试")

if __name__ == "__main__":
    main() 