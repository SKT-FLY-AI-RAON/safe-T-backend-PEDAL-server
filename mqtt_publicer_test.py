import paho.mqtt.client as mqtt
import json
import time
import random

# MQTT 브로커 정보
broker_address = "3.35.30.20"  # MQTT 브로커 주소 (Flutter와 동일한 브로커 사용)
port = 1883
topic = "pedal/topic"  # Flutter 앱에서 수신할 토픽 이름

# MQTT 클라이언트 생성
client = mqtt.Client()

# MQTT 브로커에 연결
client.connect(broker_address, port=port)

# 테스트 데이터를 MQTT로 전송하는 함수
def send_test_data():
    # 무작위로 데이터 생성
    brake_status = random.choice(["pressed", "released"])  # 임의로 페달 상태 생성
    accelerator_status = random.choice(["pressed", "released"])
    speed = random.uniform(0, 180)  # 속도는 0에서 180 km/h 사이
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간

    # 테스트용 데이터
    pedal_data = {
        'brake_status': brake_status,
        'accelerator_status': accelerator_status,
        'speed': speed,
        'timestamp': timestamp
    }

    # 데이터를 JSON 형식으로 변환
    message = json.dumps(pedal_data)

    # MQTT 메시지 전송
    client.publish(topic, message)
    print(f"Sent data: {message}")

# 주기적으로 데이터를 전송 (5초 간격)
if __name__ == "__main__":
    try:
        while True:
            send_test_data()
            time.sleep(5)  # 5초마다 데이터를 전송
    except KeyboardInterrupt:
        print("Publisher stopped.")
