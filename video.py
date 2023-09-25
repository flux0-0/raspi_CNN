import cv2
import tensorflow as tf
import numpy as np
import time
# Load mô hình nhận diện khuôn mặt từ file
save_model = tf.keras.models.load_model("/home/pi/Final/face.h5")

# Tạo bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('/home/pi/Final/haarcascade_frontalface_alt.xml')

# Mở video từ file hoặc thiết bị (điều chỉnh tham số nếu cần)
video_path = '/home/pi/Final/test/testvideo.mp4' #đường dẫn video
video_capture = cv2.VideoCapture(video_path)

#đầu vào là webcamlaptop
#video_capture = cv2.VideoCapture(0)
# Kiểm tra xem video có mở thành công hay không
if not video_capture.isOpened():
    print("Không thể mở video!")
    exit()

# Đọc và xử lý từng khung hình trong video
frame_count = 0
start_time = time.time()
while True:
    # Đọc khung hình từ video
    ret, frame = video_capture.read()
    # Kiểm tra xem video có còn khung hình hay không
    if not ret:
        break
    # Chuyển đổi khung hình sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, 1.05, 6)
    # Xử lý từng khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Cắt và chuẩn hóa khuôn mặt
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = face.reshape((1, 100, 100, 1))
        face = face / 255.0
        # Dự đoán lớp của khuôn mặt
        result = save_model.predict(face)
        final = np.argmax(result)
        # Vẽ hình chữ nhật và hiển thị nhãn lên khung hình
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = ""
        if final == 0:
            label = "Son Tung"
        elif final == 1:
            label = "Unknown"
        elif final == 2:
            label = "Unknown"
        elif final == 3:
            label = "Giang"
        elif final == 4:
            label = "Bao"
        cv2.putText(frame, label, (x+10, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Hiển thị khung hình kết quả
    cv2.imshow('Video', frame)
    # Tính FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    # Kiểm tra phím nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng tài nguyên và đóng cửa sổ
video_capture.release()
cv2.destroyAllWindows()
