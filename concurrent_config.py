#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发配置优化模块
为不同的数据集框架提供最佳的并发配置
"""

import os
from multiprocessing import cpu_count
from typing import Dict, Tuple

def get_optimal_config() -> Dict:
    """获取系统最优并发配置"""
    num_cpus = cpu_count()
    
    # 基于CPU核心数确定最优配置
    if num_cpus >= 16:
        # 高性能服务器
        config = {
            'pytorch_workers': 8,
            'ray_workers': 12,
            'hf_processes': 8,
            'batch_sizes': [8, 16, 32],
            'ray_batch_size': 16,
            'ray_num_cpus_per_task': 2,
            'ray_concurrency': 8
        }
    elif num_cpus >= 8:
        # 中等性能机器
        config = {
            'pytorch_workers': 6,
            'ray_workers': 8,
            'hf_processes': 6,
            'batch_sizes': [4, 8, 16],
            'ray_batch_size': 12,
            'ray_num_cpus_per_task': 2,
            'ray_concurrency': 6
        }
    elif num_cpus >= 4:
        # 普通机器
        config = {
            'pytorch_workers': 4,
            'ray_workers': 4,
            'hf_processes': 4,
            'batch_sizes': [4, 8],
            'ray_batch_size': 8,
            'ray_num_cpus_per_task': 1,
            'ray_concurrency': 4
        }
    else:
        # 低性能机器
        config = {
            'pytorch_workers': 2,
            'ray_workers': 2,
            'hf_processes': 2,
            'batch_sizes': [2, 4],
            'ray_batch_size': 4,
            'ray_num_cpus_per_task': 1,
            'ray_concurrency': 2
        }
    
    config['num_cpus'] = num_cpus
    return config

def get_pytorch_dataloader_config(num_workers: int = None) -> Dict:
    """获取PyTorch DataLoader的最优配置"""
    config = get_optimal_config()
    
    if num_workers is None:
        num_workers = config['pytorch_workers']
    
    return {
        'num_workers': num_workers,
        'pin_memory': True,  # 如果有GPU可用
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 4 if num_workers > 0 else 2,
        'drop_last': False,
        'timeout': 60  # 60秒超时
    }

def get_ray_config(num_workers: int = None) -> Dict:
    """获取Ray数据集的最优配置"""
    config = get_optimal_config()
    
    if num_workers is None:
        num_workers = config['ray_workers']
    
    return {
        'batch_size': config['ray_batch_size'],
        'num_cpus': config['ray_num_cpus_per_task'],
        'concurrency': min(num_workers, config['ray_concurrency']),
        'max_concurrent_tasks': num_workers
    }

def get_hf_config(num_proc: int = None) -> Dict:
    """获取HuggingFace数据集的最优配置"""
    config = get_optimal_config()
    
    if num_proc is None:
        num_proc = config['hf_processes']
    
    return {
        'num_proc': num_proc,
        'batch_size': 1000,  # HF内部批处理大小
        'writer_batch_size': 1000,
        'keep_in_memory': False,  # 对于大数据集，不保存在内存中
        'load_from_cache_file': True
    }

def print_system_info():
    """打印系统信息和推荐配置"""
    config = get_optimal_config()
    
    print("💻 系统信息和推荐配置:")
    print(f"   CPU核心数: {config['num_cpus']}")
    print(f"   PyTorch workers: {config['pytorch_workers']}")
    print(f"   Ray workers: {config['ray_workers']}")
    print(f"   HuggingFace processes: {config['hf_processes']}")
    print(f"   推荐批处理大小: {config['batch_sizes']}")
    print()

if __name__ == "__main__":
    print_system_info()
    
    print("PyTorch DataLoader配置:")
    pytorch_config = get_pytorch_dataloader_config()
    for key, value in pytorch_config.items():
        print(f"   {key}: {value}")
    
    print("\nRay数据集配置:")
    ray_config = get_ray_config()
    for key, value in ray_config.items():
        print(f"   {key}: {value}")
    
    print("\nHuggingFace数据集配置:")
    hf_config = get_hf_config()
    for key, value in hf_config.items():
        print(f"   {key}: {value}") 