import cv2
import os
import numpy as np #thư viện tính toán số học
#Dùng bộ harrcascade để phát hiện khuôn mặt trong ảnh 
detector = cv2.CascadeClassifier('/home/pi/Final/haarcascade_frontalface_alt.xml')
id=1
if id == 1: 
    print(0)
    for i in range(1,6):
        for j in range (1,21):
            filename = '/home/pi/Final/data/anh.' + str(i) + '.' + str(j) + '.jpg' #load hết các ảnh trong folder data
            frame = cv2.imread(filename)  #frame biểu diễn ảnh dưới dạng số học theo từng pixel 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #chuyển ảnh sang màu xám 
            fa = detector.detectMultiScale(gray, 1.05, 6) #phát hiện khuôn mặt trong ảnh xám  
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                if not os.path.exists('dataset'):
                    os.makedirs('dataset')
                cv2.imwrite('/home/pi/Final/dataset/anh.'  + str(i) + '.' +str(j) + '.jpg', gray[y:y+h,x:x+w])  
                #lưu khuôn mặt đã được phát hiện vào folder datashet
#hàm kiểm tra xem có ảnh nào không phát hiện ra khuôn mặt không
def check_missing_images():
    missing_images = []
    for i in range(1, 6):
        for j in range(1, 21):
            filename = f"/home/pi/Final/dataset/anh.{i}.{j}.jpg"
            if not os.path.exists(filename):
                missing_images.append(filename)
    return missing_images

missing_images = check_missing_images()
if len(missing_images) == 0:
    print("Không có ảnh nào thiếu.")
else:
    print("Các ảnh thiếu:")
    for image in missing_images:
        print(image)
