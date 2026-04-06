import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import os

def plot_history(history, model_name="model"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 绘制 Loss 
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1) 
    ax1.set_title(f'{model_name} Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # 2. 绘制 Accuracy
    ax2.plot(history.history['acc'], label='train_acc')
    ax2.plot(history.history['val_acc'], label='val_acc')
    ax2.set_ylim(0, 1.05) 
    ax2.set_title(f'{model_name} Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right')

    # 路径处理
    save_path = r'F:\复现图像生成'
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, f'{model_name}_training_plot.png')
    plt.tight_layout() # 防止标签重叠
    plt.savefig(file_path, dpi=300)
    print(f" 训练曲线（Y轴起步0）已保存: {file_path}")
    plt.close()

def evaluate_model(model, x_test, y_test, model_name="model"):
    pred = model.predict(x_test)
    y_pre = np.argmax(pred, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    print(f"\n📋 {model_name} 测试集分类报告:\n", classification_report(y_true, y_pre, digits=4))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_true, y_pre)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    save_path = r'F:\复现图像生成'
    cm_path = os.path.join(save_path, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f" 混淆矩阵已保存: {cm_path}")
    plt.close()