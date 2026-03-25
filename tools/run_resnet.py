import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# =============================================================
# 1. 【环境与路径适配】
# =============================================================
current_file_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(tools_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datas.dataset import load_pcam_dataset_streaming
from models.resnet import resnet50_baseline_model

# GPU 优化
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================
# 2. 【数据与模型加载】
# =============================================================
batch_size = 32
data_dir = os.path.join(project_root, 'pcam\\')
train_ds, val_ds, train_count, val_count = load_pcam_dataset_streaming(data_dir, batch_size=batch_size)

print(f"🚀 启动 V1 随机初始化基准实验 (策略 B+C)...")
model = resnet50_baseline_model(input_shape=(96, 96, 3), classes=2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', metrics=['acc'])

save_dir = os.path.join(project_root, 'checkpoints', 'resnet50_v1_low_benchmark')
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, 'best_low_benchmark_v1.h5')

# =============================================================
# 3. 【三剑客回调】实现自动收敛停止
# =============================================================
# A. 保存巅峰状态
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# B. 学习率自动下调
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

# C. 早停机制：监控收敛，8轮不降则自动停止，并恢复最佳权重
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    verbose=1, 
    restore_best_weights=True  # 停止后自动加载最好那一轮的参数
)

# =============================================================
# 4. 【执行训练】
# =============================================================
print(f"🔥 训练启动！若模型已收敛且 Loss 连续 5 轮不降，程序将自动停止...")
history = model.fit(
    train_ds,
    steps_per_epoch=train_count // batch_size,
    epochs=30,
    validation_data=val_ds,
    validation_steps=val_count // batch_size,
    callbacks=[checkpoint, lr_reducer, early_stop]
)

# =============================================================
# 5. 【自动化评估与图像式混淆矩阵】
# =============================================================
print("\n" + "="*60)
print("📊 训练已自动停止，正在生成最佳状态的图像化报告...")
print("="*60)

# 加载保存的巅峰模型
best_model = tf.keras.models.load_model(checkpoint_path)

y_true, y_pred = [], []
for x_batch, y_batch in val_ds.take(val_count // batch_size):
    preds = best_model.predict(x_batch, verbose=0)
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# A. 输出文本报告 (4位小数)
print("\n📋 分类报告 (Classification Report):")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], digits=4))

# B. 绘制图像式混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'],
            annot_kws={"size": 16})

plt.title('Baseline: ResNet50-V1 (Random Weights) Confusion Matrix', fontsize=16)
plt.ylabel('Ground Truth', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

# C. 自动保存图片
cm_save_path = os.path.join(save_dir, 'v1_baseline_confusion_matrix.png')
plt.savefig(cm_save_path, dpi=300)
print(f"✅ 混淆矩阵热力图已自动保存至: {cm_save_path}")

plt.show()