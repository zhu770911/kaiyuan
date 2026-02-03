import tensorflow as tf
import numpy as np
import cv2

# --- 图像预处理部分保持不变 ---
# 读取图片
img = cv2.imread('4.png')
if img is None:
    raise FileNotFoundError("Could not load image '4.png'")
# 转灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 改变尺寸
img = cv2.resize(img, (28, 28))
# 转黑底白字、归一化
img = (255 - img) / 255.0 # 明确指定浮点数除法
# 转为4维
img = img.reshape((1, 28, 28, 1))
# --- 图像预处理结束 ---

# --- 关键修改：使用 custom_objects 加载模型 ---
# 定义一个映射，告诉 Keras 用 tf.nn.softmax 替代 'softmax_v2'
try:
    # 尝试直接使用 tf.nn.softmax (较常见)
    custom_objects = {'softmax_v2': tf.nn.softmax}
    model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)
except ValueError:
    try:
        # 如果上面失败，尝试使用 tf.keras.activations.softmax
        custom_objects = {'softmax_v2': tf.keras.activations.softmax}
        model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)
    except ValueError:
         # 如果都失败，可以尝试使用 lambda 函数包装
        custom_objects = {'softmax_v2': lambda x: tf.nn.softmax(x)}
        model = tf.keras.models.load_model('model.h5', custom_objects=custom_objects)



probabilities = model.predict(img)
print(probabilities)
prediction = np.argmax(probabilities)
prediction_values = np.max(probabilities)
print('预测:  结果:{}  概率:{:.2%}'.format(prediction, prediction_values))