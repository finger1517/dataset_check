#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba JIT vs CPU基线性能对比测试
专门比较JIT编译优化的效果
"""

import os
import time
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

# Numba导入
try:
    from numba import jit, prange
    import numba as nb
    HAS_NUMBA = True
    print("✅ Numba可用，将进行JIT对比测试")
except ImportError:
    HAS_NUMBA = False
    print("❌ Numba不可用，无法进行JIT测试")
    exit(1)

# ==================== CPU基线实现（无JIT）====================

def extract_frames_cpu_baseline(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """CPU基线版本（无JIT优化）"""
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
    
    # CPU基线标准化（Python原生）
    normalized_frames = normalize_frames_python(frames)
    return normalized_frames

def normalize_frames_python(frames: np.ndarray) -> np.ndarray:
    """Python原生标准化（无JIT）"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 使用numpy向量化操作
    normalized = frames.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    
    return normalized

def normalize_frames_python_loops(frames: np.ndarray) -> np.ndarray:
    """使用Python循环的标准化（模拟最慢情况）"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = frames.astype(np.float32) / 255.0
    
    # 使用Python循环（很慢）
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            for k in range(frames.shape[2]):
                for c in range(3):
                    normalized[i, j, k, c] = (normalized[i, j, k, c] - mean[c]) / std[c]
    
    return normalized

# ==================== Numba JIT优化实现 ====================

@jit(nopython=True)
def normalize_frames_jit(frames: np.ndarray) -> np.ndarray:
    """使用Numba JIT优化的标准化"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = frames.astype(np.float32) / 255.0
    
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            for k in range(frames.shape[2]):
                for c in range(3):
                    normalized[i, j, k, c] = (normalized[i, j, k, c] - mean[c]) / std[c]
    
    return normalized

@jit(nopython=True, parallel=True)
def normalize_frames_jit_parallel(frames: np.ndarray) -> np.ndarray:
    """使用Numba JIT并行优化的标准化"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = frames.astype(np.float32) / 255.0
    
    # 使用prange进行并行处理
    for i in prange(frames.shape[0]):
        for j in range(frames.shape[1]):
            for k in range(frames.shape[2]):
                for c in range(3):
                    normalized[i, j, k, c] = (normalized[i, j, k, c] - mean[c]) / std[c]
    
    return normalized

@jit(nopython=True)
def resize_frame_jit(frame: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """使用JIT优化的简单图像缩放（最近邻）"""
    old_height, old_width = frame.shape[:2]
    
    # 预分配输出数组
    if frame.ndim == 3:
        resized = np.zeros((new_height, new_width, frame.shape[2]), dtype=frame.dtype)
    else:
        resized = np.zeros((new_height, new_width), dtype=frame.dtype)
    
    # 计算缩放比例
    y_ratio = old_height / new_height
    x_ratio = old_width / new_width
    
    for i in range(new_height):
        for j in range(new_width):
            # 最近邻插值
            y = min(int(i * y_ratio), old_height - 1)
            x = min(int(j * x_ratio), old_width - 1)
            
            if frame.ndim == 3:
                resized[i, j] = frame[y, x]
            else:
                resized[i, j] = frame[y, x]
    
    return resized

def extract_frames_jit(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """使用JIT优化的视频帧提取"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((num_frames, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 仍然使用OpenCV进行视频解码和颜色转换（这部分很难用JIT优化）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用JIT优化的缩放
            frame_resized = resize_frame_jit(frame_rgb, target_size[1], target_size[0])
            frames.append(frame_resized)
        else:
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    cap.release()
    
    frames_array = np.array(frames)
    
    # 使用JIT优化的标准化
    normalized_frames = normalize_frames_jit_parallel(frames_array)
    return normalized_frames

# ==================== 基准测试函数 ====================

def benchmark_normalization_methods(frames: np.ndarray):
    """比较不同标准化方法的性能"""
    print("\n🔬 标准化方法性能对比:")
    print("=" * 60)
    
    methods = [
        ("Python向量化", normalize_frames_python),
        ("Python循环", normalize_frames_python_loops),
        ("Numba JIT", normalize_frames_jit),
        ("Numba JIT并行", normalize_frames_jit_parallel),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"测试 {method_name}...")
        
        # 预热JIT编译（如果适用）
        if "JIT" in method_name:
            _ = method_func(frames[:1])  # 预热
        
        # 实际测试
        times = []
        for _ in range(5):  # 运行5次取平均
            start_time = time.time()
            result = method_func(frames.copy())
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[method_name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': frames.shape[0] / avg_time
        }
        
        print(f"   ⏱️  平均时间: {avg_time:.4f}±{std_time:.4f}秒")
        print(f"   🚀 处理速度: {frames.shape[0]/avg_time:.1f} 帧/秒")
    
    # 找出最快的方法
    fastest = min(results.items(), key=lambda x: x[1]['avg_time'])
    baseline = results["Python向量化"]
    
    print(f"\n🏆 最快方法: {fastest[0]}")
    print(f"📈 相比基线提升: {baseline['avg_time']/fastest[1]['avg_time']:.2f}倍")
    
    return results

def benchmark_full_pipeline(video_paths: List[str]):
    """比较完整的视频处理管道"""
    print("\n🎬 完整视频处理管道对比:")
    print("=" * 60)
    
    methods = [
        ("CPU基线", extract_frames_cpu_baseline),
        ("JIT优化", extract_frames_jit),
    ]
    
    results = {}
    test_videos = video_paths[:5]  # 测试前5个视频
    
    for method_name, method_func in methods:
        print(f"\n测试 {method_name}...")
        
        total_time = 0
        total_frames = 0
        
        for video_path in test_videos:
            print(f"   处理: {os.path.basename(video_path)}")
            
            start_time = time.time()
            frames = method_func(video_path, num_frames=16, target_size=(224, 224))
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_time += processing_time
            total_frames += frames.shape[0]
            
            print(f"      ⏱️  {processing_time:.3f}秒 ({frames.shape[0]/processing_time:.1f} 帧/秒)")
        
        avg_fps = total_frames / total_time
        
        results[method_name] = {
            'total_time': total_time,
            'total_frames': total_frames,
            'avg_fps': avg_fps
        }
        
        print(f"   📊 总计: {total_time:.3f}秒, 平均 {avg_fps:.1f} 帧/秒")
    
    # 计算性能提升
    if "CPU基线" in results and "JIT优化" in results:
        baseline_fps = results["CPU基线"]["avg_fps"]
        jit_fps = results["JIT优化"]["avg_fps"]
        improvement = jit_fps / baseline_fps
        
        print(f"\n📈 JIT优化提升: {improvement:.2f}倍")
        print(f"🎯 从 {baseline_fps:.1f} FPS 提升到 {jit_fps:.1f} FPS")
    
    return results

def plot_performance_comparison(norm_results: dict, pipeline_results: dict):
    """绘制性能对比图表"""
    try:
        # 标准化性能对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 标准化方法对比
        methods = list(norm_results.keys())
        times = [norm_results[method]['avg_time'] for method in methods]
        
        bars1 = ax1.bar(methods, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('标准化方法性能对比')
        ax1.set_ylabel('处理时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.4f}s', ha='center', va='bottom')
        
        # 完整管道对比
        if pipeline_results:
            pipeline_methods = list(pipeline_results.keys())
            fps_values = [pipeline_results[method]['avg_fps'] for method in pipeline_methods]
            
            bars2 = ax2.bar(pipeline_methods, fps_values, color=['#FF6B6B', '#45B7D1'])
            ax2.set_title('完整视频处理管道对比')
            ax2.set_ylabel('处理速度 (帧/秒)')
            
            # 添加数值标签
            for bar, fps_val in zip(bars2, fps_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{fps_val:.1f} FPS', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('jit_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 性能对比图表已保存为 jit_performance_comparison.png")
        
    except ImportError:
        print("⚠️ matplotlib不可用，跳过图表生成")

def warm_up_jit():
    """预热JIT编译"""
    print("🔥 预热JIT编译...")
    
    # 创建测试数据
    test_frames = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    
    # 预热各个JIT函数
    _ = normalize_frames_jit(test_frames)
    _ = normalize_frames_jit_parallel(test_frames)
    _ = resize_frame_jit(test_frames[0], 112, 112)
    
    print("✅ JIT预热完成")

def main():
    """主函数"""
    print("⚡ Numba JIT vs CPU基线性能对比测试")
    print("=" * 50)
    
    if not HAS_NUMBA:
        print("❌ Numba不可用，无法进行测试")
        return
    
    # 预热JIT
    warm_up_jit()
    
    # 查找视频文件
    from video_dataset_comparison import find_all_mp4_files
    video_paths = find_all_mp4_files("../video_data")
    
    if not video_paths:
        print("❌ 未找到视频文件")
        return
    
    # 1. 先测试标准化方法
    print("\n📹 加载测试数据...")
    test_video = video_paths[0]
    cap = cv2.VideoCapture(test_video)
    
    # 读取一些帧用于标准化测试
    frames = []
    for i in range(32):  # 读取32帧
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    if frames:
        frames_array = np.array(frames, dtype=np.uint8)
        norm_results = benchmark_normalization_methods(frames_array)
    else:
        print("❌ 无法读取测试帧")
        norm_results = {}
    
    # 2. 测试完整的视频处理管道
    pipeline_results = benchmark_full_pipeline(video_paths)
    
    # 3. 生成性能对比图表
    plot_performance_comparison(norm_results, pipeline_results)
    
    # 4. 总结
    print("\n📋 测试总结:")
    print("=" * 50)
    print("🔹 JIT编译在计算密集型操作（如标准化）中效果显著")
    print("🔹 对于I/O密集型操作（如视频解码），JIT提升有限")
    print("🔹 并行JIT在多核CPU上能提供额外的性能提升")
    print("🔹 首次运行JIT函数会有编译开销，后续调用速度很快")

if __name__ == "__main__":
    main() 