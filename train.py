import cv2
import numpy as np
from PIL import Image
data = [] 
label = []
for j in range (1,6):
  for i in range (1,21):
    filename = '/home/pi/Final/dataset/anh.' + str(j) + '.' + str(i) + '.jpg'  #train ảnh khuôn mặt trong dataset
    Img = cv2.imread(filename)  #đọc ảnh từ file
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY) #Chuyển ảnh sang không gian xám 
    Img = cv2.resize(src=Img, dsize=(100,100))  #Chuyển đổi kích thước ảnh thành 100x100
    Img = np.array(Img)
    data.append(Img)
    label.append(j-1) #cài đặt nhãn tương ứng với giá trị = j-1 
data1 = np.array(data)  #chuyển đổi mảng data sang mảng numpy
label = np.array(label) #chuyển đổi mảng label sang mảng numpy
data1 = data1.reshape((100,100,100,1))
X_train = data1/255 #chuẩn hóa dữ liệu: tạo kiểu dữ liệu nhỏ hơn <8bit (2^8)
from sklearn.preprocessing import LabelBinarizer #tạo nhãn dán số học
lb = LabelBinarizer()
trainY =lb.fit_transform(label)
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential #các lớp cung cấp mô hình mạng nơ-ron 
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate #các lớp để xây dựng kiến trúc mạng nơ-ron 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD #tối ưu hóa SGD cho mạng nơ-ron
Model = Sequential()
shape = (100,100, 1)
Model.add(Conv2D(32,(3,3),padding="same",input_shape=shape))
Model.add(Activation("relu"))
Model.add(Conv2D(32,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Conv2D(64,(3,3), padding="same"))
Model.add(Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2,2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation("relu"))
Model.add(Dense(5))
Model.add(Activation("softmax"))
Model.summary()
Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("start training")
Model.fit(X_train,trainY,batch_size=5,epochs=11) #huấn luyện mô hình 
Model.save("face.h5")
