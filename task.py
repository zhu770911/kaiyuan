import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载MNIST
mnist = tf.keras.datasets.mnist
# 加载MNIST数据集为训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 归一化操作
x_train, x_test = x_train / 255., x_test / 255.
# 增加维度
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
# 转换为one-hot编码
y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=10))
y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=10))
# 设置批量大小
batch_size = 64
# 载入数据为dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(batch_size).shuffle(batch_size * 10)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(batch_size)

# --- 优化内容：增加损失函数和优化器 ---
# 创建优化器实例
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# 创建损失函数实例 (适用于 one-hot 编码标签)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
# --- 优化内容结束 ---

# 输入
input_img = tf.keras.Input([28, 28, 1])
# 第一层卷积
conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation=tf.nn.relu)(input_img)
# 第二层卷积
conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu)(conv1)
# 最大池化
pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv2)
# 第三层卷积
conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu)(pool)
# flatten拉平
flat = tf.keras.layers.Flatten()(conv3)
# 全连接层
dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flat)
# 全连接层
dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(dense1)
# 指定模型的输入和输出
model = tf.keras.Model(inputs=input_img, outputs=dense2)
model.summary() # 查看网络结构

# 配置训练方法 (使用上面创建的 optimizer 和 loss_fn)
model.compile(optimizer=optimizer, # 使用自定义优化器
              loss=loss_fn,        # 使用自定义损失函数
              metrics=['accuracy'])

# 执行训练过程
model.fit(train_dataset, epochs=10)
# 模型评估
score = model.evaluate(test_dataset)
print('last score:', score)
# 保存模型
model.save('model.h5')