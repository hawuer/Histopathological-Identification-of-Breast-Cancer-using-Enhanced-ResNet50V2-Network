import os
import sys
import tensorflow as tf
import numpy as np

# 1. 【路径与环境配置】
project_root = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism'
sys.path.append(project_root)

# 从新的 dataset 加载函数导入
from datas.dataset import load_pcam_dataset_streaming, load_pcam_data
from models.se_resnet import SE_ResNet50_V1 
from utils.metrics import plot_history, evaluate_model 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 本地数据存放路径
local_data_dir = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism\pcam\\'

# GPU 优化配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU 显存按需增长已开启")

# 2. 数据流初始化 (解决 27GB 内存问题)
batch_size = 32  # 适配本地 3060 显存
print(f" 正在创建 PCam 数据流 (Batch Size: {batch_size})...")
train_ds, val_ds, train_count, val_count = load_pcam_dataset_streaming(local_data_dir, batch_size=batch_size)

# 3. 模型初始化 (适配 PCam 96x96, 2类)
print(f" 正在初始化 SE-ResNet50 (V1) - 适配 PCam...")
# 【关键修改】：input_shape 改为 (96, 96, 3)，classes 改为 2
model = SE_ResNet50_V1(input_shape=(96, 96, 3), classes=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 4. 训练回调配置 (增加学习率衰减优化)
save_dir = os.path.join(project_root, 'checkpoints', 'se_resnet_pcam_streaming')
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = os.path.join(save_dir, 'best_se_resnet_pcam.h5')
checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True
)

# 针对 PCam 数据量大的特点，加入学习率衰减，防止震荡
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# 5. 执行训练
print(f" 开始执行 SE-ResNet50 (V1) PCam 实验...")
history = model.fit(
    train_ds,
    steps_per_epoch=train_count // batch_size,
    epochs=30,
    validation_data=val_ds,
    validation_steps=val_count // batch_size,
    callbacks=[checkpoint, lr_reducer]
)

# 6. 结果产出
print("\n 正在生成 SE-ResNet50 评估报告...")
plot_history(history, model_name="se_resnet50_pcam")

# 加载测试集评估 (同样需要使用新版 load_pcam_data)
_, _, (x_test, y_test) = load_pcam_data(data_dir=local_data_dir)
evaluate_model(model, x_test, y_test, model_name="se_resnet50_pcam")

print(f" 实验完成！模型已保存至: {checkpoint_path}")