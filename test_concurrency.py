#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发能力测试脚本
快速测试各个数据集框架的并发性能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from concurrent_config import get_optimal_config, print_system_info
from video_dataset_comparison import *
from multiprocessing import cpu_count
import time

def test_pytorch_concurrency(video_paths: List[str]) -> List[Dict]:
    """测试PyTorch不同并发配置的性能"""
    results = []
    config = get_optimal_config()
    
    print("🔥 测试PyTorch并发性能...")
    
    # 测试不同的worker数量
    worker_counts = [1, 2, 4, config['pytorch_workers']]
    batch_sizes = [4, 8]
    
    for num_workers in worker_counts:
        for batch_size in batch_sizes:
            if num_workers > cpu_count():
                continue
                
            print(f"   测试配置: {num_workers} workers, batch_size={batch_size}")
            
            start_time = time.time()
            
            dataset = PyTorchVideoDataset(video_paths) 