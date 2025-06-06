#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示批处理(batch)在Ray数据集中的重要作用
"""

import time
import ray
import numpy as np
from typing import Dict, List
import os

def find_all_mp4_files(base_path: str) -> List[str]:
    """递归查找所有mp4文件"""
    mp4_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

# 模拟视频处理函数
def simulate_video_processing(video_path: str) -> np.ndarray:
    """模拟视频处理（添加一些延迟来模拟实际处理时间）"""
    time.sleep(0.1)  # 模拟100ms的处理时间
    return np.random.rand(16, 224, 224, 3)

# ==================== 方式1: 不使用batch（逐个处理）====================
def process_single_video(item: Dict) -> Dict:
    """不使用batch，逐个处理视频"""
    video_path = item['video_path']
    
    print(f"🔄 处理单个视频: {os.path.basename(video_path)}")
    frames = simulate_video_processing(video_path)
    
    return {
        'frames': frames,
        'video_path': video_path,
        'video_name': os.path.basename(video_path)
    }

# ==================== 方式2: 使用batch（批量处理）====================
def process_video_batch(batch: Dict) -> Dict:
    """使用batch，批量处理视频"""
    video_paths = batch['video_path']
    
    print(f"📦 处理批次: {len(video_paths)} 个视频")
    
    processed_batch = {
        'frames': [],
        'video_path': [],
        'video_name': []
    }
    
    # 在一个函数调用中处理多个视频
    for video_path in video_paths:
        frames = simulate_video_processing(video_path)
        
        processed_batch['frames'].append(frames)
        processed_batch['video_path'].append(video_path)
        processed_batch['video_name'].append(os.path.basename(video_path))
    
    return processed_batch

# ==================== 方式3: 使用batch + 并发（最优）====================
from concurrent.futures import ThreadPoolExecutor

def process_video_batch_concurrent(batch: Dict) -> Dict:
    """使用batch + 并发，最优处理方式"""
    video_paths = batch['video_path']
    
    print(f"🚀 并发处理批次: {len(video_paths)} 个视频")
    
    processed_batch = {
        'frames': [],
        'video_path': [],
        'video_name': []
    }
    
    # 使用线程池并发处理批次中的视频
    with ThreadPoolExecutor(max_workers=min(4, len(video_paths))) as executor:
        futures = {executor.submit(simulate_video_processing, path): path for path in video_paths}
        
        for future in futures:
            video_path = futures[future]
            frames = future.result()
            
            processed_batch['frames'].append(frames)
            processed_batch['video_path'].append(video_path)
            processed_batch['video_name'].append(os.path.basename(video_path))
    
    return processed_batch

def compare_processing_methods(video_paths: List[str]):
    """比较不同处理方式的性能"""
    # 使用较少的视频进行演示
    test_paths = video_paths[:20]  
    
    print(f"📊 比较不同处理方式 (使用 {len(test_paths)} 个视频)")
    print("=" * 60)
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    # 方式1: 不使用batch
    print("\n🔄 方式1: 不使用batch (逐个处理)")
    start_time = time.time()
    
    data = [{'video_path': path} for path in test_paths]
    ds1 = ray.data.from_items(data)
    result1 = ds1.map(process_single_video)  # 每个视频一次函数调用
    
    # 强制执行所有任务
    processed_count1 = 0
    for item in result1.iter_rows():
        processed_count1 += 1
    
    time1 = time.time() - start_time
    print(f"   ⏱️  总时间: {time1:.2f}秒")
    print(f"   📝 函数调用次数: {len(test_paths)} 次")
    print(f"   🎯 平均每次调用: {time1/len(test_paths):.3f}秒")
    
    # 方式2: 使用batch
    print("\n📦 方式2: 使用batch (批量处理)")
    start_time = time.time()
    
    batch_size = 4
    ds2 = ray.data.from_items(data)
    result2 = ds2.map_batches(process_video_batch, batch_size=batch_size)
    
    processed_count2 = 0
    batch_count = 0
    for batch in result2.iter_batches():
        batch_count += 1
        processed_count2 += len(batch['frames'])
    
    time2 = time.time() - start_time
    print(f"   ⏱️  总时间: {time2:.2f}秒")
    print(f"   📝 函数调用次数: {batch_count} 次")
    print(f"   📦 每批处理: {batch_size} 个视频")
    print(f"   🎯 平均每次调用: {time2/batch_count:.3f}秒")
    
    # 方式3: 使用batch + 并发
    print("\n🚀 方式3: 使用batch + 并发 (最优)")
    start_time = time.time()
    
    ds3 = ray.data.from_items(data)
    result3 = ds3.map_batches(process_video_batch_concurrent, batch_size=batch_size)
    
    processed_count3 = 0
    batch_count3 = 0
    for batch in result3.iter_batches():
        batch_count3 += 1
        processed_count3 += len(batch['frames'])
    
    time3 = time.time() - start_time
    print(f"   ⏱️  总时间: {time3:.2f}秒")
    print(f"   📝 函数调用次数: {batch_count3} 次")
    print(f"   📦 每批处理: {batch_size} 个视频")
    print(f"   🎯 平均每次调用: {time3/batch_count3:.3f}秒")
    
    # 性能对比
    print("\n📈 性能对比:")
    print("=" * 60)
    print(f"方式1 (逐个): {time1:.2f}秒 - 基准")
    print(f"方式2 (批次): {time2:.2f}秒 - 提升 {((time1-time2)/time1*100):.1f}%")
    print(f"方式3 (并发): {time3:.2f}秒 - 提升 {((time1-time3)/time1*100):.1f}%")
    
    print("\n💡 关键洞察:")
    print(f"🔹 batch减少了函数调用开销: {len(test_paths)} 次 → {batch_count} 次")
    print(f"🔹 并发进一步利用了CPU资源，实现了真正的并行处理")
    print(f"🔹 Ray的分布式特性在batch模式下更有效")

def demonstrate_batch_structure():
    """演示batch的数据结构"""
    print("\n🔍 Batch数据结构演示:")
    print("=" * 50)
    
    # 模拟原始数据
    video_paths = ['/path/video1.mp4', '/path/video2.mp4', '/path/video3.mp4', '/path/video4.mp4']
    
    print("📝 原始数据 (单个项目):")
    for i, path in enumerate(video_paths):
        print(f"   item_{i}: {{'video_path': '{path}'}}")
    
    print("\n📦 Batch数据 (batch_size=4):")
    batch = {
        'video_path': video_paths
    }
    print(f"   batch: {batch}")
    
    print("\n🔄 处理后的Batch数据:")
    processed_batch = {
        'frames': [f"frames_from_{os.path.basename(path)}" for path in video_paths],
        'video_path': video_paths,
        'video_name': [os.path.basename(path) for path in video_paths]
    }
    
    for key, values in processed_batch.items():
        print(f"   {key}: {values}")
    
    print("\n💡 关键优势:")
    print("🔹 一次函数调用处理多个视频")
    print("🔹 减少Ray任务调度开销")
    print("🔹 更好的内存局部性")
    print("🔹 可以在batch内部使用并发")

def main():
    """主函数"""
    print("🎬 Batch处理演示")
    
    # 演示数据结构
    demonstrate_batch_structure()
    
    # 查找视频文件
    video_paths = find_all_mp4_files("../video_data")
    
    if video_paths:
        # 比较处理方式
        compare_processing_methods(video_paths)
    else:
        print("\n⚠️ 未找到视频文件，使用模拟数据演示")
        # 使用模拟路径
        mock_paths = [f"/mock/video_{i}.mp4" for i in range(20)]
        compare_processing_methods(mock_paths)
    
    # 清理
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main() 