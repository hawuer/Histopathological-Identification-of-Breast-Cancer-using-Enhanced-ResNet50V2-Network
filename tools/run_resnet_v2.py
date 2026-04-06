import os
import sys
import tensorflow as tf
import numpy as np

# 1. 【路径配置】
project_root = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism'
sys.path.append(project_root)

# 导入流式加载函数和模型
from datas.dataset import load_pcam_dataset_streaming, load_pcam_data
from models.resnet_v2 import ResNet50V2
from utils.metrics import plot_history, evaluate_model 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 数据存放路径
local_data_dir = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism\pcam\\'

# GPU 配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(" GPU 显存按需增长已开启")

# 2. 数据流初始化 (解决 27GB 内存问题)
batch_size = 32
print(f" 正在创建 PCam 数据流 (Batch Size: {batch_size})...")
train_ds, val_ds, train_count, val_count = load_pcam_dataset_streaming(local_data_dir, batch_size=batch_size)

# 3. 模型构建 (适配 96x96, 2类)
print(" 正在启动 ResNet50V2 (Baseline) 适配 PCam 训练...")
# 确保 input_shape 和 classes 正确
model = ResNet50V2(input_shape=(96, 96, 3), classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# 4. 回调函数 (加入学习率衰减，解决 Loss 震荡)
save_dir = os.path.join(project_root, 'checkpoints', 'resnet_v2_pcam_streaming')
os.makedirs(save_dir, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(save_dir, 'best_resnet_v2_pcam.h5'),
    monitor='val_acc',
    save_best_only=True,
    verbose=1
)

# 新增：当 val_loss 3代不下降时，学习率减半，能有效解决你刚才看到的 800+ Loss 的问题
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# 5. 执行训练
history = model.fit(
    train_ds,
    steps_per_epoch=train_count // batch_size,
    epochs=30,
    validation_data=val_ds,
    validation_steps=val_count // batch_size,
    callbacks=[checkpoint, lr_reducer]
)

# 6. 评估与出图
print("\n 正在生成 ResNet50V2 评估报告...")
plot_history(history, model_name="resnet50_v2_pcam")

# 加载测试集进行评估
_, _, (x_test, y_test) = load_pcam_data(data_dir=local_data_dir)
evaluate_model(model, x_test, y_test, model_name="resnet50_v2_pcam")

print(f" 实验完成！结果保存在: {save_dir}")