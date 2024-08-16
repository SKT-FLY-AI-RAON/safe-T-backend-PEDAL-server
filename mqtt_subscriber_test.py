import paho.mqtt.client as mqtt

# MQTT 브로커 주소 및 포트 설정
mqtt_broker = "3.35.30.20"  # 서버에서 사용하는 브로커 주소
mqtt_port = 1883  # 기본 MQTT 포트
mqtt_topic = "pedal/topic"  # 서버에서 사용하는 토픽 이름

# 메시지를 받을 때 호출되는 콜백 함수
def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()} on topic {message.topic}")

# MQTT 클라이언트 생성
mqtt_client = mqtt.Client()

# 메시지 수신 시 호출될 콜백 함수 설정
mqtt_client.on_message = on_message

# 브로커에 연결
mqtt_client.connect(mqtt_broker, mqtt_port)

# 특정 토픽 구독
mqtt_client.subscribe(mqtt_topic)

# 구독 대기 (계속 실행)
print(f"Subscribed to topic: {mqtt_topic}")
mqtt_client.loop_forever()
