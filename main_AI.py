from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re

load_dotenv()
app = FastAPI()

class WordOnly(BaseModel):
    word: str

summery_template = """
너는 단어 사전 작성 전문가야.
아래 단어가 어떤 실체를 의미하는지 파악하고, 그 실체가 속하는 가장 적절한 카테고리 1개 이상 선택해라. 여러개여도 된다.

반드시 해당 단어가 의미하는 '하나의 실체'만 골라서 설명하라.
반드시 니가 가장 먼저 추론한 카테고리와 관련된 카테고리끼리 연관지어 그것만 설명해야 한다.
예를 들어 '배'라는 단어는 음식, 과일 카테고리와 교통, 운송수단에도 다 적절하지만 네가 음식 쪽으로 먼저 추론 헸다면 무조건 음식과 관련된 카테고리를 생성하고 그것을 중심으로 설명해라.
또한 하나의 단어에 여러 뜻이 있을 때 똑같은 뜻만 계속 출력하려하지 말고 때에 따라서 다양하게 출력하도록 해라. 예를 들어 배에 운송수단의 의미와 음식의 의미와 동물의 배 등의 의미가 있으면 실행 될 때마다 골고루 출력 되도록 하여라.
그리고 제발 카테고리 추론 했을 때 가장 먼저 추론한거만 그거만 출력하여라.
- 입력 단어: {information}

출력 형식은 반드시 다음과 같아야 한다:

카테고리: ["카테고리1", "카테고리2", ...]
설명: (그 단어에 대한 설명을 자연스럽고 자세히 작성하라)

단, 출력 형식이 반드시 아래처럼 JSON 스타일로 나와야 한다:
카테고리: ["예시1", "예시2"]
설명: 여기 설명을 길게 써도 되고 짧게 써도 된다. 단, 카테고리 이름이나 단어 이름을 반복해서 말하지 마라.

설명은 무조건 한국어로 작성해라.
"""

prompt = PromptTemplate(
    input_variables=["information"],
    template=summery_template
)
llm = ChatOpenAI(temperature=1, model_name="gpt-4o")
chain = prompt | llm | StrOutputParser()

@app.post("/define")
async def define_word(data: WordOnly):
    try:
        result_text = chain.invoke({"information": data.word.strip()})

        category_match = re.search(r'카테고리:\s*\[(.*?)\]', result_text, re.DOTALL)
        description_match = re.search(r'설명:\s*(.*)', result_text, re.DOTALL)

        if not category_match or not description_match:
            raise HTTPException(status_code=500, detail="출력 형식이 올바르지 않습니다.")

        categories_raw = category_match.group(1)
        categories = [cat.strip().strip('"') for cat in categories_raw.split(",")]

        description = description_match.group(1).strip()

        return {
            "categories": categories,
            "description": description
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"에러 발생: {str(e)}")
