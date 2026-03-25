import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# 1. 路径适配
# =============================================================
current_file_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(utils_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# =============================================================
# 2. 定位训练过程数据 (history.csv)
# =============================================================
# 确保该路径与您 SE-ResNet50V2 训练保存的日志路径一致
save_dir = os.path.join(project_root, 'checkpoints', 'se_resnet50v2_stable_final')
history_path = os.path.join(save_dir, 'history.csv')

print(f"✅ 正在尝试读取历史数据: {history_path}")

if not os.path.exists(history_path):
    print(f"❌ 错误：未在 {save_dir} 下找到 history.csv 文件！")
    print("提示：请确认训练脚本中是否使用了 CSVLogger 记录数据，或手动将控制台输出的数值存入该路径。")
    sys.exit()

# 读取真实的实验数据
df = pd.read_csv(history_path)

# =============================================================
# 3. 绘图展示 (单独绘制 Loss 和 Accuracy)
# =============================================================
# 设置学术绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 左图：Loss 曲线 ---
ax1.plot(df['loss'], label='训练损失 (Train Loss)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
ax1.plot(df['val_loss'], label='验证损失 (Val Loss)', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
ax1.set_title('SE-ResNet50V2 训练与验证损失收敛图', fontsize=14, fontweight='bold')
ax1.set_xlabel('迭代轮次 (Epochs)', fontsize=12)
ax1.set_ylabel('损失值 (Loss)', fontsize=12)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# --- 右图：Accuracy 曲线 (核心修改) ---
ax2.plot(df['acc'], label='训练准确率 (Train Acc)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
ax2.plot(df['val_acc'], label='验证准确率 (Val Acc)', color='#ff7f0e', linewidth=2, marker='s', markersize=4)

# 【关键步骤】：强制 Y 轴从 0 开始
ax2.set_ylim(0, 1.05)

ax2.set_title('SE-ResNet50V2 训练与验证准确率变化图', fontsize=14, fontweight='bold')
ax2.set_xlabel('迭代轮次 (Epochs)', fontsize=12)
ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.6)

# 自动调整布局
plt.tight_layout()

# 保存图像
final_save_path = os.path.join(utils_dir, 'se_v2_final_training_plot_v2.png')
plt.savefig(final_save_path, dpi=300, bbox_inches='tight')

print(f"✅ 包含真实过程数据的曲线图已保存至: {final_save_path}")
plt.show()