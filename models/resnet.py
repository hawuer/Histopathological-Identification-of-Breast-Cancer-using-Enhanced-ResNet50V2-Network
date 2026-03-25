import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import ResNet50 

def resnet50_baseline_model(input_shape=(96, 96, 3), classes=2):
    """
    严格受限的标准 ResNet50 (V1) 基准模型
    策略 B: 禁用预训练权重 (weights=None)，强制从随机状态开始学习
    策略 C: 压缩分类器 (Dense 128) 并设置极高 Dropout (0.7)
    """
    # 核心：使用随机初始化的 V1 卷积基
    conv_base = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    
    model = Sequential([
        conv_base,
        GlobalAveragePooling2D(), 
        # 限制隐藏层神经元数量，增加泛化难度
        Dense(128, activation='relu'),
        Dropout(0.7), 
        Dense(classes, activation='softmax')
    ])
    
    # 随机初始化下，必须解冻全量参数进行学习
    conv_base.trainable = True
    return model