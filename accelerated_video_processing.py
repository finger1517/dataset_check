#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加速视频处理模块
使用GPU、Numba JIT和多线程优化视频帧提取
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
    from numba import jit, cuda
    import numba as nb
    HAS_NUMBA = True
    print("✅ Numba可用，将使用JIT编译")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba不可用，将使用普通Python")

# FFmpeg相关导入（更快的视频解码）
try:
    import ffmpeg
    HAS_FFMPEG = True
    print("✅ FFmpeg-python可用，将使用FFmpeg解码")
except ImportError:
    HAS_FFMPEG = False
    print("⚠️ FFmpeg-python不可用，将使用OpenCV")

# ==================== GPU加速实现 ====================

def extract_frames_gpu(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """使用GPU加速的视频帧提取"""
    if not HAS_CUPY:
        return extract_frames_cpu_optimized(video_path, num_frames, target_size)
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # 计算均匀采样的帧索引
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # 预分配GPU内存
        frames_gpu = cp.zeros((num_frames, target_size[0], target_size[1], 3), dtype=cp.uint8)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # CPU上调整大小（OpenCV在CPU上更快）
                frame_resized = cv2.resize(frame, target_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # 转移到GPU进行后续处理
                frame_gpu = cp.asarray(frame_rgb)
                frames_gpu[i] = frame_gpu
        
        cap.release()
        
        # 将结果转回CPU
        return cp.asnumpy(frames_gpu)
        
    except Exception as e:
        print(f"GPU处理失败，回退到CPU: {e}")
        return extract_frames_cpu_optimized(video_path, num_frames, target_size)

@cuda.jit if HAS_NUMBA else lambda x: x
def normalize_frames_gpu(frames, mean, std):
    """GPU上的帧标准化"""
    i, j, k, c = cuda.grid(4)
    if i < frames.shape[0] and j < frames.shape[1] and k < frames.shape[2] and c < frames.shape[3]:
        frames[i, j, k, c] = (frames[i, j, k, c] / 255.0 - mean[c]) / std[c]

# ==================== Numba JIT优化实现 ====================

@jit(nopython=True, parallel=True) if HAS_NUMBA else lambda x: x
def resize_frame_numba(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """使用Numba JIT优化的帧大小调整"""
    # 注意：这是一个简化的双线性插值实现
    # 在实际应用中，OpenCV的resize通常更快更准确
    h_old, w_old = frame.shape[:2]
    h_new, w_new = target_size
    
    # 预分配输出数组
    if frame.ndim == 3:
        resized = np.zeros((h_new, w_new, frame.shape[2]), dtype=frame.dtype)
    else:
        resized = np.zeros((h_new, w_new), dtype=frame.dtype)
    
    # 计算缩放比例
    x_ratio = w_old / w_new
    y_ratio = h_old / h_new
    
    for i in range(h_new):
        for j in range(w_new):
            # 最近邻插值（简化版）
            x = int(j * x_ratio)
            y = int(i * y_ratio)
            
            # 边界检查
            x = min(x, w_old - 1)
            y = min(y, h_old - 1)
            
            if frame.ndim == 3:
                resized[i, j] = frame[y, x]
            else:
                resized[i, j] = frame[y, x]
    
    return resized

@jit(nopython=True, parallel=True) if HAS_NUMBA else lambda x: x
def normalize_frames_numba(frames: np.ndarray) -> np.ndarray:
    """使用Numba JIT优化的帧标准化"""
    # ImageNet标准化参数
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 转换为float并标准化
    normalized = frames.astype(np.float32) / 255.0
    
    for i in range(frames.shape[0]):  # 遍历帧
        for j in range(frames.shape[1]):  # 高度
            for k in range(frames.shape[2]):  # 宽度
                for c in range(3):  # RGB通道
                    normalized[i, j, k, c] = (normalized[i, j, k, c] - mean[c]) / std[c]
    
    return normalized

# ==================== FFmpeg优化实现 ====================

def extract_frames_ffmpeg(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """使用FFmpeg进行更快的视频解码"""
    if not HAS_FFMPEG:
        return extract_frames_cpu_optimized(video_path, num_frames, target_size)
    
    try:
        # 获取视频信息
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        total_frames = int(video_info['nb_frames']) if 'nb_frames' in video_info else 1000
        duration = float(video_info['duration']) if 'duration' in video_info else 10.0
        
        # 计算时间戳
        timestamps = np.linspace(0, duration, num_frames)
        
        frames = []
        for timestamp in timestamps:
            try:
                # 使用FFmpeg直接跳转到指定时间戳
                out = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .filter('scale', target_size[0], target_size[1])
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
                
                # 解析原始RGB数据
                frame = np.frombuffer(out[0], np.uint8).reshape(target_size[1], target_size[0], 3)
                frames.append(frame)
                
            except Exception as e:
                # 如果某个帧提取失败，添加零帧
                frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        
        return np.array(frames)
        
    except Exception as e:
        print(f"FFmpeg处理失败，回退到OpenCV: {e}")
        return extract_frames_cpu_optimized(video_path, num_frames, target_size)

# ==================== CPU多线程优化实现 ====================

def extract_single_frame(args):
    """提取单个帧的函数，用于多进程"""
    video_path, frame_idx, target_size = args
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_resized = cv2.resize(frame, target_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

def extract_frames_multiprocess(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """使用多进程并行提取帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames == 0:
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # 计算帧索引
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 准备参数
    args = [(video_path, idx, target_size) for idx in frame_indices]
    
    # 使用进程池
    # with ProcessPoolExecutor(max_workers=min(num_frames, mp.cpu_count())) as executor:
    with ProcessPoolExecutor(max_workers=1) as executor:
        frames = list(executor.map(extract_single_frame, args))
    
    return np.array(frames)

# ==================== CPU优化实现（基线）====================

def extract_frames_cpu_optimized(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """CPU优化版本的视频帧提取"""
    cap = cv2.VideoCapture(video_path)
    
    # 优化：设置缓冲区大小
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # 计算帧索引
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 预分配数组
    frames = np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 直接写入预分配的数组
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames[i] = frame_rgb
    
    cap.release()
    return frames

# ==================== 智能方法选择 ====================

class AcceleratedVideoProcessor:
    """智能视频处理器，自动选择最佳加速方法"""
    
    def __init__(self):
        self.methods = []
        self._setup_methods()
    
    def _setup_methods(self):
        """设置可用的处理方法"""
        # 按优先级排序
        if HAS_CUPY:
            self.methods.append(('GPU', extract_frames_gpu))
        
        if HAS_FFMPEG:
            self.methods.append(('FFmpeg', extract_frames_ffmpeg))
        
        # 多进程方法（对于大视频文件效果好）
        self.methods.append(('MultiProcess', extract_frames_multiprocess))
        
        # CPU优化方法（基线）
        self.methods.append(('CPU', extract_frames_cpu_optimized))
        
        print(f"可用的加速方法: {[name for name, _ in self.methods]}")
    
    def benchmark_methods(self, video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)):
        """基准测试所有可用方法"""
        print(f"\n🔬 基准测试视频: {os.path.basename(video_path)}")
        print("=" * 60)
        
        results = {}
        
        for method_name, method_func in self.methods:
            try:
                print(f"测试 {method_name}...")
                start_time = time.time()
                
                frames = method_func(video_path, num_frames, target_size)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                results[method_name] = {
                    'time': processing_time,
                    'fps': num_frames / processing_time,
                    'frames_shape': frames.shape,
                    'success': True
                }
                
                print(f"   ✅ {method_name}: {processing_time:.3f}秒 ({num_frames/processing_time:.1f} FPS)")
                
            except Exception as e:
                results[method_name] = {
                    'time': float('inf'),
                    'fps': 0,
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ {method_name}: 失败 - {e}")
        
        # 找出最快的方法
        successful_methods = {k: v for k, v in results.items() if v['success']}
        if successful_methods:
            fastest = min(successful_methods.items(), key=lambda x: x[1]['time'])
            print(f"\n🏆 最快方法: {fastest[0]} ({fastest[1]['fps']:.1f} FPS)")
        
        return results
    
    def extract_frames(self, video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224), 
                      method: Optional[str] = None) -> np.ndarray:
        """提取视频帧，自动或手动选择方法"""
        if method:
            # 使用指定方法
            method_func = next((func for name, func in self.methods if name.lower() == method.lower()), None)
            if method_func:
                return method_func(video_path, num_frames, target_size)
            else:
                print(f"未找到方法 {method}，使用默认方法")
        
        # 使用第一个可用方法（通常是最快的）
        for method_name, method_func in self.methods:
            try:
                return method_func(video_path, num_frames, target_size)
            except Exception as e:
                print(f"{method_name} 失败: {e}")
                continue
        
        # 如果所有方法都失败，返回零数组
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)

# ==================== 测试和演示 ====================

def benchmark_all_methods(video_paths: List[str]):
    """对多个视频文件进行全面基准测试"""
    processor = AcceleratedVideoProcessor()
    
    print("🚀 开始全面性能测试")
    print("=" * 80)
    
    # 使用不同大小的视频进行测试
    test_videos = video_paths[:10] if len(video_paths) >= 3 else video_paths
    
    all_results = {}
    
    for video_path in test_videos:
        print(f"\n📹 测试视频: {os.path.basename(video_path)}")
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"   分辨率: {width}x{height}, 帧数: {frame_count}, FPS: {fps:.1f}")
        
        results = processor.benchmark_methods(video_path)
        all_results[video_path] = results
    
    # 汇总结果
    print("\n📊 性能汇总:")
    print("=" * 80)
    
    method_totals = {}
    for video_results in all_results.values():
        for method_name, result in video_results.items():
            if result['success']:
                if method_name not in method_totals:
                    method_totals[method_name] = []
                method_totals[method_name].append(result['fps'])
    
    for method_name, fps_list in method_totals.items():
        avg_fps = np.mean(fps_list)
        print(f"🎯 {method_name}: 平均 {avg_fps:.1f} FPS")
    
    return all_results

def main():
    """主函数"""
    print("⚡ 加速视频处理测试")
    
    # 查找视频文件
    from video_dataset_comparison import find_all_mp4_files
    video_paths = find_all_mp4_files("../video_data")
    
    if not video_paths:
        print("❌ 未找到视频文件")
        return
    
    # 运行基准测试
    results = benchmark_all_methods(video_paths)
    
    # 演示使用
    processor = AcceleratedVideoProcessor()
    
    print("\n🎬 实际使用示例:")
    test_video = video_paths[0]
    
    start_time = time.time()
    frames = processor.extract_frames(test_video, num_frames=16, target_size=(224, 224))
    end_time = time.time()
    
    print(f"✅ 提取 {frames.shape[0]} 帧，耗时 {end_time - start_time:.3f}秒")
    print(f"📐 帧尺寸: {frames.shape}")

if __name__ == "__main__":
    main() 