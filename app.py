import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import threading
import time
import cv2
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
from collections import Counter
from datetime import datetime  # 타임스탬프 추가
import paho.mqtt.client as mqtt

app = Flask(__name__)

# CORS 설정 추가
CORS(app)

# 전역 변수로 상태 플래그 설정
is_running = False  # 페달 모델이 실행 중인지 여부를 나타냄
video_writer = None  # 영상 저장을 위한 VideoWriter 객체
frame_count = 0  # 프레임 카운터
predicted_labels = []  # 예측된 결과를 저장할 리스트

# MQTT 설정
mqtt_broker = "3.35.30.20"  # MQTT 브로커 주소 (예: 'localhost' 또는 실제 브로커 주소)
mqtt_port = 1883  # MQTT 브로커 포트 (기본적으로 1883)
mqtt_topic = "pedal/topic"  # 전송할 토픽 이름

# MQTT 클라이언트 설정
mqtt_client = mqtt.Client()

# MQTT 브로커에 연결
def connect_mqtt():
    try:
        mqtt_client.connect(mqtt_broker, mqtt_port)
        print(f"Connected to MQTT Broker at {mqtt_broker}:{mqtt_port}")
    except Exception as e:
        print(f"Failed to connect to MQTT Broker: {e}")

# MQTT로 메시지 전송 함수
def send_mqtt_message(message):
    try:
        mqtt_client.publish(mqtt_topic, message)
        print(f"Sent message to MQTT topic {mqtt_topic}: {message}")
    except Exception as e:
        print(f"Failed to send message via MQTT: {e}")

# 연결 설정 (앱 시작 시 연결)
connect_mqtt()

# Swagger 설정
SWAGGER_URL = '/swagger'  # Swagger UI 접근 경로
API_URL = '/static/swagger.json'  # Swagger JSON 파일 경로

# Swagger UI 설정
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI 경로
    API_URL,      # Swagger JSON 파일 경로
    config={      # Swagger UI 설정
        'app_name': "페달 분류 API"  # API 이름
    }
)

# Swagger UI Blueprint 등록
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# 이미지 전처리 클래스
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
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

# 모델 초기화 및 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load('data_0813_left_best_model_wts_18_epoch50.pth', map_location=device))
model.eval()

# 이미지 전처리 파이프라인 설정
resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = ImageTransform(resize, mean, std)

# 클래스 이름과 숫자 할당
class_names = ['acc_push', 'brake_push', 'acc', 'brake']
class_dict = {class_name: i for i, class_name in enumerate(class_names)}  # 클래스에 숫자 매기기

# IP Webcam URL (IP 주소를 확인 후 사용)
ip_webcam_url = "http://172.23.251.156:8080/video"

# 타임스탬프 형식 설정
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_pedal_classification():
    global is_running, video_writer, frame_count, predicted_labels
    cap = cv2.VideoCapture(ip_webcam_url)

    # 프레임 크기 자동 추출
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # 영상 저장을 위한 VideoWriter 객체 생성 (코덱 확인 및 자동화)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
    output_video_path = f'output_video_{get_timestamp()}.avi'  # 타임스탬프를 이용한 파일명 생성
    fps = 30  # FPS 설정
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    while is_running:
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
            predicted_labels.append(class_dict[label])  # 예측된 클래스의 숫자를 리스트에 추가

            # 예측 결과 출력 (실시간 출력)
            print(f"Predicted class: {label}")

            # 프레임에 예측된 클래스 라벨을 추가
            cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 영상 저장 (프레임을 파일에 기록)
        if video_writer is not None:
            video_writer.write(frame)

        # 30프레임(1초)마다 결과를 확인
        if frame_count % 30 == 0:
            # 1초 동안의 예측 결과 출력
            print(f"1초 동안의 예측 결과: {predicted_labels}")

            # 가장 많이 나온 클래스 찾기
            most_common_class = Counter(predicted_labels).most_common(1)[0]
            most_common_class_name = class_names[most_common_class[0]]
            print(f"가장 많이 나온 클래스: {most_common_class_name} (등장 횟수: {most_common_class[1]})")

            # 1초 동안 결과가 'acc_push' 또는 'brake_push'일 경우 MQTT로 전송
            if most_common_class_name in ['acc_push', 'brake_push']:
                send_mqtt_message(most_common_class_name)  # MQTT 메시지 전송

            # 리스트 초기화
            predicted_labels = []

        time.sleep(1 / fps)  # FPS 맞춰 대기 (1초에 30프레임)

    cap.release()
    if video_writer is not None:
        video_writer.release()  # 비디오 파일 저장 완료
    print("페달 분류 모델 중단 및 영상 저장 완료")


# 페달 분류 시작 API
@app.route('/start-pedal-model', methods=['POST'])
def start_pedal_model():
    global is_running

    if not is_running:
        is_running = True
        send_mqtt_message("start")
        threading.Thread(target=run_pedal_classification).start()
        return jsonify({"status": "페달 분류 모델 실행 시작"}), 200
    else:
        return jsonify({"status": "페달 분류 모델 이미 실행 중"}), 200

# 페달 분류 중단 API
@app.route('/stop-pedal-model', methods=['POST'])
def stop_pedal_model():
    global is_running

    if is_running:
        is_running = False
        send_mqtt_message("end")
        return jsonify({"status": "페달 분류 모델 중단"}), 200
    else:
        return jsonify({"status": "페달 분류 모델이 실행 중이 아님"}), 200

# 기본 경로 설정 ("/")
@app.route('/')
def index():
    return "페달 분류 API 서버가 실행 중입니다."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001)

