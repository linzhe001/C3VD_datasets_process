import os
import zipfile
import shutil
import re
import rarfile  # 需要安装: pip install rarfile
import argparse

def extract_archives(source_dir, temp_dir):
    """解压源目录中的所有压缩包到临时目录"""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # 获取所有压缩文件
    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if file.endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    print(f"正在解压 {file}...")
                    zip_ref.extractall(temp_dir)
                    print(f"{file} 解压完成")
            except Exception as e:
                print(f"解压 {file} 时出错: {e}")
        
        elif file.endswith('.rar'):
            try:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    print(f"正在解压 {file}...")
                    rar_ref.extractall(temp_dir)
                    print(f"{file} 解压完成")
            except Exception as e:
                print(f"解压 {file} 时出错: {e}")

def move_files(temp_dir, target_dir):
    """将临时目录中的所有文件移动到目标目录"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # 获取临时目录中的所有项目
    for item in os.listdir(temp_dir):
        source_path = os.path.join(temp_dir, item)
        target_path = os.path.join(target_dir, item)
        
        try:
            print(f"正在移动 {item} 到目标目录...")
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    # 如果目标是目录，则合并内容
                    for sub_item in os.listdir(source_path):
                        sub_source = os.path.join(source_path, sub_item)
                        sub_target = os.path.join(target_path, sub_item)
                        shutil.move(sub_source, sub_target)
                    os.rmdir(source_path)  # 移除空目录
                else:
                    # 如果目标是文件，则替换
                    os.remove(target_path)
                    shutil.move(source_path, target_path)
            else:
                # 目标不存在，直接移动
                shutil.move(source_path, target_path)
            print(f"{item} 移动完成")
        except Exception as e:
            print(f"移动 {item} 时出错: {e}")

def delete_unwanted_files(target_dir):
    """删除目标目录中所有包含指定模式的文件"""
    patterns = ["_color", "_flow", "_normals", "_occlusion"]
    files_deleted = 0
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if any(pattern in file for pattern in patterns):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    files_deleted += 1
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除 {file_path} 时出错: {e}")
    
    print(f"共删除了 {files_deleted} 个不需要的文件")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='解压缩文件并清理特定文件')
    parser.add_argument('--source_dir', required=True, help='源压缩包所在目录的路径')
    parser.add_argument('--target_dir', required=True, help='解压后文件的目标目录路径')
    args = parser.parse_args()
    
    # 设置目录路径
    source_dir = args.source_dir
    target_dir = args.target_dir
    temp_dir = os.path.join(source_dir, "temp_extracted")
    
    # 执行操作
    print("开始处理...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"临时目录: {temp_dir}")
    
    extract_archives(source_dir, temp_dir)
    move_files(temp_dir, target_dir)
    delete_unwanted_files(target_dir)
    
    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
        print("临时目录已清理")
    except Exception as e:
        print(f"清理临时目录时出错: {e}")
    
    print("所有操作已完成!")

if __name__ == "__main__":
    main()