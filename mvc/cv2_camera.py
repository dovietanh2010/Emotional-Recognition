import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")


print("Bắt đầu chương trình...")

cap = cv2.VideoCapture(0)
print("Đã mở camera.")

if not cap.isOpened():
    print("Không thể mở camera.")
else:
    print("Camera mở thành công!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể lấy khung hình từ camera.")
        break

    results = model(frame)
    frame = cv2.flip(frame, 0)
    annotated_frame = results[0].plot()
    cv2.imshow('Camera Thời Gian Thực', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Đã đóng kết nối và hủy cửa sổ.")
