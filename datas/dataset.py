import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical # 补全引用

# =============================================================
# 1. 生成器函数：解决内存溢出与维度不匹配
# =============================================================
def pcam_generator(x_path, y_path, batch_size):
    """
    按批次从 H5 文件读取数据，避免一次性加载 27GB 内存。
    """
    with h5py.File(x_path, 'r') as f_x, h5py.File(y_path, 'r') as f_y:
        # 获取 H5 文件内部的对象引用
        x_data = f_x['x']
        y_data = f_y['y']
        num_samples = x_data.shape[0]
        
        while True:
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                
                # 1. 读取影像并归一化 (0-255 -> 0.0-1.0)
                batch_x = x_data[i:end].astype('float32') / 255.0
                
                # 2. 【核心修复】：使用 np.squeeze 移除多余的维度
                # 解决此前 (32, 1, 1, 2) 导致的 InvalidArgumentError
                batch_y_raw = np.squeeze(y_data[i:end])
                
                # 3. 转换为 One-hot 编码 (适配二分类)
                batch_y = tf.keras.utils.to_categorical(batch_y_raw, num_classes=2)
                
                yield batch_x, batch_y

# =============================================================
# 2. 数据集加载函数：构建 TensorFlow 数据流水线
# =============================================================
def load_pcam_dataset_streaming(data_dir, batch_size=32):
    """
    创建 tf.data.Dataset 对象，支持边读边练，节省内存。
    """
    # 拼接本地 H5 文件路径
    train_x_path = data_dir + 'camelyonpatch_level_2_split_train_x.h5'
    train_y_path = data_dir + 'camelyonpatch_level_2_split_train_y.h5'
    val_x_path = data_dir + 'camelyonpatch_level_2_split_valid_x.h5'
    val_y_path = data_dir + 'camelyonpatch_level_2_split_valid_y.h5'
    
    # 预先获取总样本数以计算 steps_per_epoch
    with h5py.File(train_x_path, 'r') as f:
        train_count = f['x'].shape[0]
    with h5py.File(val_x_path, 'r') as f:
        val_count = f['x'].shape[0]

    # 定义输出数据的格式 (Signature)
    # None 表示 batch_size 可以自动适配
    output_sig = (
        tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )

    # 构建训练集 Dataset
    train_ds = tf.data.Dataset.from_generator(
        lambda: pcam_generator(train_x_path, train_y_path, batch_size),
        output_signature=output_sig
    ).prefetch(tf.data.AUTOTUNE) # 利用 CPU 预取加速

    # 构建验证集 Dataset
    val_ds = tf.data.Dataset.from_generator(
        lambda: pcam_generator(val_x_path, val_y_path, batch_size),
        output_signature=output_sig
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_count, val_count

# =============================================================
# 3. 辅助函数：仅用于测试集评估
# =============================================================
def load_pcam_data(data_dir):
    """
    用于评估阶段一次性加载测试集 (测试集通常较小，内存可承受)。
    """
    test_x_path = data_dir + 'camelyonpatch_level_2_split_test_x.h5'
    test_y_path = data_dir + 'camelyonpatch_level_2_split_test_y.h5'
    
    with h5py.File(test_x_path, 'r') as f_x, h5py.File(test_y_path, 'r') as f_y:
        x_test = f_x['x'][:].astype('float32') / 255.0
        # 同样需要 squeeze 处理防止评估报错
        y_test = np.squeeze(f_y['y'][:])
        y_test = to_categorical(y_test, num_classes=2)
        
    return None, None, (x_test, y_test)