# coding: utf-8
from tkinter.filedialog import askopenfilename
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D,Dropout
from keras.layers import Flatten,Dense
import numpy as np
from PIL import Image
np.random.seed(0)
import cv2
import numpy as np
import read_images_labels as read
import os



# 忽略硬件加速的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
PROVINCE_START = 1000
image_size = 32
LABEL_FILE = '../labels.txt'
IMAGE_SIZE = 32  # 图片默认大小
num_classes = 10  # 图片种类数
data_dir = '../train_images/'
labels = {0: 'sansejin', 1: 'baxianhua', 2: 'bianhua', 3: 'lihua', 4: 'qianniuhua',
          5: 'qiangwei', 6: 'xunyicao', 7: 'hudielan', 8: 'jidanhua', 9: 'yuanwei'}


# 读取图片文件
def image_read(filename):
    img_list = []
    img = Image.open(filename)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)
    img_list.append(img)
    img = np.array(img_list)
    return img


def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)



class CardPredictor:
    def __del__(self):
        self.save_traindata()

    def train_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=48, kernel_size=(3, 3), input_shape=(image_size, image_size, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=82, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.20))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))
        print(self.model.summary())
        if os.path.exists("../flower10model1.h5"):
            self.model.load_weights("../flower10model1.h5")
            print("模型加载成功")
        else:
            x_train, y_train = read.read_images_labels(data_dir=data_dir, batch_size=10000)
            x_train_one = x_train * (1. / 255) - 0.5
            self.model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
            train_history = self.model.fit(x=x_train_one, y=y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

    def save_traindata(self):
        if not os.path.exists("../flower10model1.h5"):
            self.model.save_weights("../flower10model1.h5")

    def predict(self, img):
        if type(img) == type(""):
            img = image_read(img)

        # pic_hight, pic_width = img.shape[:2]
        # if pic_width > MAX_WIDTH:
        #     resize_rate = MAX_WIDTH / pic_width
        #     img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)

        img = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3])
        img = img * (1. / 255) - 0.5
        resp = self.model.predict_classes(img)[0]
        label = labels[resp]
        return label


if __name__ == '__main__':
    c = CardPredictor()
    c.train_model()
    pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
    img = imreadex(pic_path)
    r = c.predict(img)
    print(r)
