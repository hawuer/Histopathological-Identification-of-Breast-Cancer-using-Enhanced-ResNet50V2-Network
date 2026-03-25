import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================
# 1. 自动适配根目录逻辑 (解决 ModuleNotFoundError)
# =============================================================
current_file_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(tools_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以安全导入项目内模块
from datas.dataset import load_pcam_dataset_streaming

# =============================================================
# 2. 路径配置 (指向 ResNet50V2 Baseline 实验路径)
# =============================================================
model_path = os.path.join(project_root, 'checkpoints', 'resnet_v2_pcam_streaming', 'best_resnet_v2_pcam.h5')
data_dir = os.path.join(project_root, 'pcam\\')

print(f"✅ 项目根目录已识别: {project_root}")
print(f"⏳ 正在加载最佳模型 (Epoch 15)...")

if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件，请检查路径: {model_path}")
    sys.exit()

model = tf.keras.models.load_model(model_path)
batch_size = 32
_, val_ds, _, val_count = load_pcam_dataset_streaming(data_dir, batch_size=batch_size)

# =============================================================
# 3. 执行评估
# =============================================================
y_true, y_pred = [], []
print("🚀 正在执行流式预测并生成混淆矩阵...")

for x_batch, y_batch in val_ds.take(val_count // 32):
    preds = model.predict(x_batch, verbose=0)
    y_true.extend(np.argmax(y_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 计算分类报告和混淆矩阵
report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], digits=4)
cm = confusion_matrix(y_true, y_pred)

# =============================================================
# 4. 输出文本结果
# =============================================================
print("\n" + "="*60)
print("📊 ResNet50V2 Baseline 最佳状态 (Epoch 15) 论文数据")
print("="*60)
print(report)
print("-" * 60)
print(f"📋 混淆矩阵数值: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
print("="*60)

# =============================================================
# 5. 【核心新增】绘制可视化混淆矩阵图
# =============================================================
plt.figure(figsize=(10, 8))
# 使用 'Blues' 配色，符合学术审美
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'],
            annot_kws={"size": 16})

plt.title('Confusion Matrix: ResNet50V2', fontsize=16)
plt.ylabel('Actual Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)

# 保存图像至脚本所在目录
save_path = os.path.join(tools_dir, 'resnet_v2_baseline_cm.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 混淆矩阵热力图已保存至: {save_path}")

plt.show()