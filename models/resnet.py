import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import ResNet50 

def resnet50_baseline_model(input_shape=(96, 96, 3), classes=2):
    
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