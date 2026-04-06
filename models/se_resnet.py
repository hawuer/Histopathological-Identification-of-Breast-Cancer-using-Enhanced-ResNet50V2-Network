import tensorflow as tf
from tensorflow.keras import layers, models

def se_block(input_tensor, ratio=16):

    channel = input_tensor.shape[-1]
    
    # Squeeze: 全局平均池化，将空间维度压缩为 1x1
    x = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation: 瓶颈结构全连接层
    # 第一个 Dense 降维
    x = layers.Dense(channel // ratio, activation='relu')(x)
    # 第二个 Dense 升维回原始通道数，使用 sigmoid 将权重映射到 0-1
    x = layers.Dense(channel, activation='sigmoid')(x)
    
    # Scale: 通道加权
    x = layers.Reshape((1, 1, channel))(x)
    return layers.Multiply()([input_tensor, x])

def resnet_v1_block(input_tensor, filters, kernel_size=3, stride=1, use_se=True):

    shortcut = input_tensor
    
    # 第一层卷积
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 第二层卷积
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 【核心】：在相加之前插入 SE 注意力模块
    if use_se:
        x = se_block(x)
        
    # 处理 Shortcut 维度不匹配的情况
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    # 相加并在相加后进行 ReLU
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def SE_ResNet50_V1(input_shape=(96, 96, 3), classes=2):

    img_input = layers.Input(shape=input_shape)
    
    # 入口层
    x = layers.Conv2D(64, (7, 7), strides=1, padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 堆叠残差块
    # Stage 1
    x = resnet_v1_block(x, 64)
    x = resnet_v1_block(x, 64)
    
    # Stage 2
    x = resnet_v1_block(x, 128, stride=2)
    x = resnet_v1_block(x, 128)
    
    # Stage 3
    x = resnet_v1_block(x, 256, stride=2)
    x = resnet_v1_block(x, 256)
    
    # 输出层
    x = layers.GlobalAveragePooling2D()(x)
    
    # 展平后的全连接层
    x = layers.Dense(512, activation='relu')(x)
    
    # 加入 Dropout(0.5) ，解决测试集准确率上不去的问题
    x = layers.Dropout(0.5)(x)
    
    # 最终分类层
    x = layers.Dense(classes, activation='softmax')(x)
    
    model = models.Model(img_input, x, name='SE_ResNet50_V1_PCam')
    return model

if __name__ == "__main__":
    # 调试测试
    model = SE_ResNet50_V1()
    model.summary()
    print("SE-ResNet50 V1 适配版构建成功")