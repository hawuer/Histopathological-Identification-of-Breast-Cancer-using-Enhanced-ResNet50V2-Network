import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =============================================================
# 1. 【路径修复】自动锁定根目录，解决 ModuleNotFoundError
# =============================================================
current_file_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(utils_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入流式加载模块
from datas.dataset import load_pcam_dataset_streaming

# =============================================================
# 2. 路径配置 (自动适配绝对路径)
# =============================================================
model_path = os.path.join(project_root, 'checkpoints', 'se_resnet50v2_stable_final', 'best_se_v2_model.h5')
data_dir = os.path.join(project_root, 'pcam\\')

print(f"✅ 项目根目录: {project_root}")
print(f"⏳ 正在加载最佳模型 (Epoch 8)...")

if not os.path.exists(model_path):
    print(f"❌ 错误：在路径 {model_path} 下未找到模型！")
    sys.exit()

model = tf.keras.models.load_model(model_path)
_, val_ds, _, val_count = load_pcam_dataset_streaming(data_dir, batch_size=32)

# =============================================================
# 3. 执行评估 (流式预测)
# =============================================================
y_true, y_pred = [], []
print("🚀 正在执行全量预测，准备绘制混淆矩阵...")

for x_batch, y_batch in val_ds.take(val_count // 32):
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 计算混淆矩阵数值
cm = confusion_matrix(y_true, y_pred)

# =============================================================
# 4. 绘图展示 (无文字模式，仅输出图像)
# =============================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'],
            annot_kws={"size": 16})

plt.title('Confusion Matrix: SE-ResNet50V2', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

# 自动保存图像到当前文件夹
save_path = os.path.join(utils_dir, 'se_resnet_v2_cm.png')
plt.savefig(save_path, dpi=300)
print(f"✅ 混淆矩阵图已保存至: {save_path}")

plt.show()