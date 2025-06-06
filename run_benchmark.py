#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行基准测试脚本
"""

import os
import sys
import time
from pathlib import Path

def main():
    print("🎬 视频数据集处理效能比较工具")
    print("=" * 50)
    
    # 检查视频数据目录
    video_data_path = Path("../video_data")
    if not video_data_path.exists():
        print(f"❌ 视频数据目录不存在: {video_data_path.absolute()}")
        print("请确保video_data目录存在并包含MP4文件")
        return
    
    # 查找视频文件
    video_files = list(video_data_path.rglob("*.mp4"))
    if not video_files:
        print(f"❌ 在 {video_data_path.absolute()} 中未找到MP4文件")
        return
    
    print(f"📁 找到 {len(video_files)} 个视频文件")
    print(f"📍 视频数据路径: {video_data_path.absolute()}")
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 完整比较测试 (推荐)")
    print("2. 仅测试PyTorch数据集")
    print("3. 仅测试Ray数据集")
    print("4. 仅测试HuggingFace数据集")
    print("5. 退出")
    
    try:
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 开始完整比较测试...")
            os.system("python video_dataset_comparison.py")
            
        elif choice == "2":
            print("\n🔥 开始PyTorch数据集测试...")
            os.system("python pytorch_data.py")
            
        elif choice == "3":
            print("\n⚡ 开始Ray数据集测试...")
            os.system("python ray_data.py")
            
        elif choice == "4":
            print("\n🤗 开始HuggingFace数据集测试...")
            os.system("python huggingface_data.py")
            
        elif choice == "5":
            print("👋 退出程序")
            return
            
        else:
            print("❌ 无效选择，请重新运行程序")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")

if __name__ == "__main__":
    main() 