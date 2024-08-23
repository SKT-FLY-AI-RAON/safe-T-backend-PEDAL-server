import openai
from dotenv import load_dotenv
import os

# 환경 변수에서 API 키 로드
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

openai.api_key = OPENAI_API_KEY

# 차량별 스크립트 데이터
car_scripts = {
    "현대 아반떼": "브레이크를 밟고 기어를 중립으로 전환한 후, 시동 버튼을 길게 눌러 엔진을 끄세요.",
    "기아 K5": "브레이크를 밟고, 기어를 N으로 전환하세요. 브레이크 오버라이드 시스템이 작동할 수 있습니다.",
    "BMW X5": "브레이크를 밟으며 기어를 N으로 바꾸고, 전자식 주차 브레이크를 사용하세요.",
    "벤츠 E 클래스": "중립 기어로 전환 후, 전자식 파킹 브레이크를 사용해 차량을 멈추세요.",
    "폭스바겐 골프": "브레이크를 밟고 기어를 중립으로 전환하세요. 전자식 브레이크를 사용할 수 있습니다.",
    "포드 익스플로러": "기어를 중립으로 전환한 후, 브레이크를 강하게 밟으세요.",
    "렉서스 RX": "브레이크를 계속 밟고, 기어를 중립으로 전환하세요. 브레이크 오버라이드 시스템을 활용하세요.",
    "혼다 어코드": "기어를 중립으로 전환하고, 브레이크를 강하게 밟으세요.",
    "테슬라 모델 3": "기어를 중립으로 전환하고, 브레이크를 강하게 밟으세요. 필요시 긴급 전원을 차단하세요.",
    "쉐보레 말리부": "브레이크를 밟고 기어를 N으로 전환하세요."
}

def gpt_interaction(make_and_model):
    """
    차량 모델에 따라 GPT로부터 급발진 상황 대처법을 얻는 함수
    :param make_and_model: 차량 모델 이름 (예: "기아 K5")
    :return: GPT에서 응답한 대처법 (문자열)
    """
    # 차량 모델에 맞는 스크립트 찾기
    if make_and_model in car_scripts:
        guide_script = car_scripts[make_and_model]
    else:
        guide_script = "해당 차량 모델에 대한 정보가 없습니다. 차량 정보를 확인해주세요."

    # GPT 프롬프트 구성
    prompt = (
        f"차량 모델: {make_and_model}\n"
        # f"차량이 급발진 상황에서 속도를 줄이지 못하는 경우, 아래의 대처 방법을 따르세요:\n\n"
        f"{guide_script}\n\n"
        # "다른 말 붙이지 말고 위 스크립트의 명령어만 **그대로** 출력하세요. 어떤 내용도 생략하거나 요약하거나 추가하지 마세요."
        "위 스크립트의 차량 모델은 출력하지 말고 명령어만 **그대로** 출력하세요. 어떤 내용도 생략하거나 요약하거나 추가하지 마세요."
    )

    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 급발진 상황에서 사용자를 지원하는 서포터입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )

    return response.choices[0]['message']['content'].strip()

# 함수 사용 예시
if __name__ == "__main__":
    car_model = "기아 K5"  # 사용자가 입력한 차량 모델
    guide_response = gpt_interaction(car_model)
    print(f"{guide_response}")