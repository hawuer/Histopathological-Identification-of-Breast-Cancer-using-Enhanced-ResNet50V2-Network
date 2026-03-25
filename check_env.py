import os
import platform
import tensorflow as tf
import psutil
import nvidia_smi

def print_experiment_config():
    print("="*20 + " 实验硬件配置 (Hardware) " + "="*20)
    # 操作系统
    print(f"操作系统: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    
    # CPU 信息
    print(f"中央处理器 (CPU): {platform.processor()}")
    print(f"物理核心数: {psutil.cpu_count(logical=False)}, 逻辑核心数: {psutil.cpu_count(logical=True)}")
    print(f"内存总量: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")

    # GPU 信息 (通过 TensorFlow 和 nvidia-smi)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"检测到 GPU 设备: {gpu}")
        
        try:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            device_name = nvidia_smi.nvmlDeviceGetName(handle)
            print(f"显卡型号: {device_name.decode('utf-8') if isinstance(device_name, bytes) else device_name}")
            print(f"显存总量: {round(info.total / (1024**2), 2)} MB")
        except:
            print("显卡详细信息通过 nvidia-smi 获取失败，请检查驱动。")
    else:
        print("未检测到可用 GPU，请检查 CUDA 配置。")

    print("\n" + "="*20 + " 实验软件环境 (Software) " + "="*20)
    print(f"Python 版本: {platform.python_version()}")
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 检查是否支持 CUDA
    print(f"GPU 是否可用 (TF): {tf.test.is_gpu_available()}")
    print(f"CUDA 版本 (内建): {tf.sysconfig.get_build_info().get('cuda_version', 'N/A')}")
    print(f"cuDNN 版本 (内建): {tf.sysconfig.get_build_info().get('cudnn_version', 'N/A')}")

if __name__ == "__main__":
    print_experiment_config()