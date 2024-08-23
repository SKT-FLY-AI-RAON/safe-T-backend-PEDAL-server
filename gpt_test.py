import openai
from dotenv import load_dotenv
import os
from openai import OpenAI


# 환경변수에서 API 키 로드
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

client = OpenAI(
    api_key = OPENAI_API_KEY,
)

messages=[]

def ask_gpt_for_solution():
    # 테스트할 프롬프트
    # prompt = "차량의 브레이크가 반복적으로 작동하지 않는 문제가 발생했습니다. 이 상황에서 운전자가 취할 수 있는 가장 안전한 대처 방법을 간단히 한글로 설명해주세요."
    # prompt = "너는 급발진상황에서 어떻게 해야할지 알려주는 서포터야.  지금부터 입력하는 상황에 맞게 어떻게 해야하는지 알려줘 극박한 상황에서 파악할수 있게 짧은 명령어로 한번에 하나의 지시만 해줘"
    prompt = (
        "너는 급발진 상황에서 운전자가 안전하게 대처할 수 있도록 도와주는 서포터야. "
        "각 단계마다 간결하고 명확한 명령어를 하나씩 제시해줘. 번호 매기지 말고 명령어만 말해줘. "
        "예시로는 '브레이크를 강하게 밟아라', '비상등을 켜라', '엔진을 중립으로 변경해라'와 같이 해줘."
    )


    try:
        messages.append(
            {
                "role": "system",
                "content": prompt,
            },
        )

        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # print("OpenAI 응답:", response)

        message = response.choices[0].message.content
        return message

    except Exception as e:
        print(f"GPT API 호출 중 오류 발생: {e}")
        return "GPT 호출 실패"

# 테스트 실행
if __name__ == '__main__':
    response = ask_gpt_for_solution()
    print(f"GPT로부터 받은 응답: {response}")
