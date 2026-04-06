import h5py
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

def pcam_generator(x_path, y_path, batch_size):

    with h5py.File(x_path, 'r') as f_x, h5py.File(y_path, 'r') as f_y:

        x_data = f_x['x']
        y_data = f_y['y']
        num_samples = x_data.shape[0]
        
        while True:
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)
                
                # 读取影像并归一化 (0-255 -> 0.0-1.0)
                batch_x = x_data[i:end].astype('float32') / 255.0
                
                # np.squeeze 移除多余的维度
                batch_y_raw = np.squeeze(y_data[i:end])
                
                # 转换为 One-hot 编码 (适配二分类)
                batch_y = tf.keras.utils.to_categorical(batch_y_raw, num_classes=2)
                
                yield batch_x, batch_y

def load_pcam_dataset_streaming(data_dir, batch_size=32):

    train_x_path = data_dir + 'camelyonpatch_level_2_split_train_x.h5'
    train_y_path = data_dir + 'camelyonpatch_level_2_split_train_y.h5'
    val_x_path = data_dir + 'camelyonpatch_level_2_split_valid_x.h5'
    val_y_path = data_dir + 'camelyonpatch_level_2_split_valid_y.h5'
    
    with h5py.File(train_x_path, 'r') as f:
        train_count = f['x'].shape[0]
    with h5py.File(val_x_path, 'r') as f:
        val_count = f['x'].shape[0]

    output_sig = (
        tf.TensorSpec(shape=(None, 96, 96, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: pcam_generator(train_x_path, train_y_path, batch_size),
        output_signature=output_sig
    ).prefetch(tf.data.AUTOTUNE) 

    val_ds = tf.data.Dataset.from_generator(
        lambda: pcam_generator(val_x_path, val_y_path, batch_size),
        output_signature=output_sig
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_count, val_count

def load_pcam_data(data_dir):

    test_x_path = data_dir + 'camelyonpatch_level_2_split_test_x.h5'
    test_y_path = data_dir + 'camelyonpatch_level_2_split_test_y.h5'
    
    with h5py.File(test_x_path, 'r') as f_x, h5py.File(test_y_path, 'r') as f_y:
        x_test = f_x['x'][:].astype('float32') / 255.0
        # 需要 squeeze 处理防止评估报错
        y_test = np.squeeze(f_y['y'][:])
        y_test = to_categorical(y_test, num_classes=2)
        
    return None, None, (x_test, y_test)