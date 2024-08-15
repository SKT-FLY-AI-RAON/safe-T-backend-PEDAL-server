import cv2

# IP Webcam의 URL 입력 (스마트폰에서 IP와 포트를 확인한 후 아래에 입력)
url = "http://172.30.1.43:8080/video"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream from the IP address.")
else:
    print("Video stream opened successfully.")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame. Exiting...")
        break
    
    if frame is None or frame.size == 0:
        print("Error: Received an empty frame.")
        break
    
    # 프레임을 이미지 파일로 저장
    cv2.imwrite(f"frame_{frame_count}.jpg", frame)
    frame_count += 1

cap.release()