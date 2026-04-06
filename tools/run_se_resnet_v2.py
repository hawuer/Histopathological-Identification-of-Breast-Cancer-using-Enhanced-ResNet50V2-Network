import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 1. 路径与环境配置
project_root = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism'
sys.path.append(project_root)

# 确保使用标准精度，防止出现之前的 nan 崩溃
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')

# 导入你的模块
from datas.dataset import load_pcam_dataset_streaming, load_pcam_data 
from models.se_resnet_v2 import SE_ResNet50V2
from utils.metrics import plot_history, evaluate_model 

local_data_dir = r'E:\Workspace\Medical Image Classification Model Based on SE-ResNet50V2 Attention Mechanism\pcam\\'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(" 实验环境准备就绪 (Float32 + 显存按需增长)")

# 2. 数据流创建 (Batch Size: 32)
batch_size = 32
print(f" 正在创建 PCam 数据流...")
train_ds, val_ds, train_count, val_count = load_pcam_dataset_streaming(local_data_dir, batch_size=batch_size)

# 3. 模型初始化
print(f" 正在加载 SE-ResNet50V2 模型...")
model = SE_ResNet50V2(input_shape=(96, 96, 3), classes=2)

# 采用你确定的稳健学习率 5e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['acc']
)

# 4. 训练回调配置
save_dir = os.path.join(project_root, 'checkpoints', 'se_resnet50v2_stable_final')
os.makedirs(save_dir, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(save_dir, 'best_se_v2_model.h5'), 
    monitor='val_acc', verbose=1, save_best_only=True
)

# 动态学习率调整
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

# 5. 执行训练 (30 Epochs)
print(f" 开始训练...")
history = model.fit(
    train_ds,
    steps_per_epoch=train_count // batch_size,
    epochs=30,
    validation_data=val_ds,
    validation_steps=val_count // batch_size,
    callbacks=[checkpoint, lr_reducer]
)

# 6. 核心修改：安全评估逻辑 (避开 MemoryError)
print("\n 正在保存训练曲线图...")
plot_history(history, model_name="se_v2_final")

print(" 正在执行流式评估以生成分类报告...")

# 直接利用 val_ds 进行分批预测
y_true = []
y_pred = []

# 遍历验证集数据流进行预测
for x_batch, y_batch in val_ds.take(val_count // batch_size):
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

from sklearn.metrics import classification_report, confusion_matrix
print("\n 最终评估报告 (Validation Set):")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

print(f" 所有任务已完成！最佳模型保存在: {save_dir}")