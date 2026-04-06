import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, 
                                     BatchNormalization, GlobalAveragePooling2D, 
                                     Dropout, Conv2D, MaxPooling2D)

# 1. 核心注意力模块 (SE Block)
def se_block(input_tensor, reduction_ratio=8):
    num_channels = input_tensor.shape[-1]
    # Squeeze: 压缩空间维度
    se = GlobalAveragePooling2D()(input_tensor)
    # Excitation: 学习通道间关系 (降低压缩比至 8 以保留更多病理细节)
    se = Dense(num_channels // reduction_ratio, activation='relu')(se)
    se = Dense(num_channels, activation='sigmoid')(se)
    # Scale: 重新加权
    se = layers.Reshape((1, 1, num_channels))(se)
    return layers.Multiply()([input_tensor, se])

# 2. ResNetV2 核心组件 (包含残差结构)
def block_v2_se(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    # Pre-activation: V2 结构的精华，BN->ReLU->Conv
    preact = BatchNormalization(name=name + '_preact_bn')(x)
    preact = Activation('relu', name=name + '_preact_relu')(preact)

    # Shortcut 路径控制
    if conv_shortcut:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    # 主分支卷积层
    x = Conv2D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)

    # 在 Add 之前融入 SE 模块
    x = se_block(x, reduction_ratio=8)

    # 残差相加
    x = Add(name=name + '_out')([shortcut, x])
    return x

# 3. 最终集成模型 (包含数据增强与 Dropout)
def SE_ResNet50V2(input_shape=(96, 96, 3), classes=2):
    img_input = Input(shape=input_shape)
    
    # 针对 PCam 数据集的旋转无关性，加入随机翻转和旋转
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(img_input)
    x = layers.RandomRotation(0.2, name="aug_rotation")(x)
    # 针对病理切片加入轻微的对比度增强
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)
    
    # 入口层 (Stem)
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = Conv2D(64, 7, strides=2, name='conv1_conv')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    # 堆叠残差 Stage
    def stack_v2_se(x, filters, blocks, stride1=2, name=None):
        x = block_v2_se(x, filters, conv_shortcut=True, name=name + '_block1')
        for i in range(2, blocks):
            x = block_v2_se(x, filters, name=name + '_block' + str(i))
        x = block_v2_se(x, filters, stride=stride1, name=name + '_block' + str(blocks))
        return x

    x = stack_v2_se(x, 64, 3, name='conv2')
    x = stack_v2_se(x, 128, 4, name='conv3')
    x = stack_v2_se(x, 256, 6, name='conv4')
    x = stack_v2_se(x, 512, 3, stride1=1, name='conv5')

    # 末端处理
    x = BatchNormalization(name='post_bn')(x)
    x = Activation('relu', name='post_relu')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    # 【Dropout 思路：分类器防守】
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5, name='dropout_final')(x) 
    
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = models.Model(img_input, x, name="SE_ResNet50V2_with_Aug")
    return model