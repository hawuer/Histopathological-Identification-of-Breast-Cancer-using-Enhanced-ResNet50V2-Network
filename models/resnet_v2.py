import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_v2_block(input_tensor, filters, kernel_size=3, stride=1):
    """
    标准 ResNet V2 残差块 (BN -> ReLU -> Conv)
    """
    # V2 的核心：先做 BN 和 ReLU (预激活)
    preact = layers.BatchNormalization()(input_tensor)
    preact = layers.Activation('relu')(preact)
    
    shortcut = input_tensor
    
    # 处理 Shortcut 的维度匹配
    # 注意：在 V2 中，如果需要下采样，应该从预激活后的信号 (preact) 引出 shortcut
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(preact)

    # 第一层卷积
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(preact)
    
    # 第二层：继续预激活逻辑
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    
    # 直接相加
    x = layers.Add()([x, shortcut])
    return x

def ResNet50V2(input_shape=(96, 96, 3), classes=2):
    """
    构建适配 PCam 的 ResNet50V2 模型
    """
    img_input = layers.Input(shape=input_shape)
    
    # 入口层
    x = layers.Conv2D(64, (7, 7), strides=1, padding='same')(img_input)
    
    # 堆叠残差块 (维持你原有的层数，确保作为轻量级 Baseline)
    # Stage 1
    x = resnet_v2_block(x, 64)
    x = resnet_v2_block(x, 64)
    
    # Stage 2
    x = resnet_v2_block(x, 128, stride=2)
    x = resnet_v2_block(x, 128)
    
    # Stage 3
    x = resnet_v2_block(x, 256, stride=2)
    x = resnet_v2_block(x, 256)
    
    # 最后一步：BN + ReLU (V2 标准收尾)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 输出层
    x = layers.GlobalAveragePooling2D()(x)
    
    # 【关键修改】：加入 Dropout 抑制过拟合，解决之前 20% 分差的问题
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(classes, activation='softmax')(x)
    
    model = models.Model(img_input, x, name='ResNet50V2_PCam')
    return model

# 调试用
if __name__ == "__main__":
    m = ResNet50V2()
    m.summary()