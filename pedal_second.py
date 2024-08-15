import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
from collections import Counter

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='val'):
        return self.data_transform[phase](img)

# 모델 로드 및 수정
model = models.resnet18(pretrained=True)  # 사전 학습된 ResNet18 모델 사용
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4개의 클래스로 분류

# 모델 초기화 및 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load('data_0813_left_best_model_wts_18_epoch50.pth', map_location=device))  # 가중치 로드
model.eval()  # 추론 모드

# 이미지 전처리 파이프라인
resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = ImageTransform(resize, mean, std)

# 클래스 이름과 숫자 할당
# acc_push: 0, brake_push:1, acc:2, brake:3
class_names = ['acc_push', 'brake_push', 'acc', 'brake']
class_dict = {class_name: i for i, class_name in enumerate(class_names)}  # 클래스에 숫자 매기기

# IP Webcam URL (IP 주소를 확인 후 사용)
ip_webcam_url = "http://172.30.1.43:8080/video"
cap = cv2.VideoCapture(ip_webcam_url)

# 리스트에 결과 저장
results = []

# 1초에 30프레임씩 결과 저장
fps = 30  # 1초에 30프레임 처리
frame_count = 0

# 출력 동영상 설정 (동영상 저장 추가)
output_video_path = 'ip_webcam_test_result_18.avi'  # 출력 동영상 파일 경로
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # VideoWriter 객체 생성

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame. Exiting...")
        break

    frame_count += 1

    # 프레임 전처리
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img, phase='val').unsqueeze(0).to(device)

    # 모델 예측
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds[0]]

        # 숫자로 변환하여 리스트에 추가
        results.append(class_dict[label])

        # 프레임에 예측 결과 라벨 표시
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 30프레임이 처리될 때마다 리스트에서 가장 많이 등장한 클래스를 출력
    if frame_count % 30 == 0:
        # 1초 동안의 결과 출력
        print(f"1초 동안의 결과 (30프레임): {results}")

        # 가장 많이 등장한 클래스 출력
        most_common_class = Counter(results).most_common(1)[0]
        print(f"가장 많이 등장한 클래스: {class_names[most_common_class[0]]}, 등장 횟수: {most_common_class[1]}")

        # 결과 리스트 초기화
        results = []

    # 실시간으로 프레임 확인
    cv2.imshow('Video Stream', frame)

    # 처리된 프레임을 출력 동영상 파일에 작성 (동영상 저장 추가)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업이 끝난 후 리소스 해제
cap.release()
out.release()  # VideoWriter 객체 해제 (동영상 저장 완료)
cv2.destroyAllWindows()

print(f"동영상 저장 완료: {output_video_path}")
