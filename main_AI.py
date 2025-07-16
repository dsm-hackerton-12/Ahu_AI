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

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from langchain_core.output_parsers import StrOutputParser
# from langchain import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# import os
#
# # 1. 환경 변수 로드 및 FastAPI 앱 초기화
# load_dotenv()
# app = FastAPI()
#
# # 2. 요청 바디 모델 정의
# class DefinitionRequest(BaseModel):
#     information: str
#     cats: list[str]  # ["음식", "기술", ...]
#
# # 3. LangChain 프롬프트 정의
# summery_template = """
# 너는 단어 사전 작성 전문가야. 명심해
# 다음은 사용자가 입력한 단어와 관련된 카테고리 목록이야:
#
# - 단어: {information}
# - 카테고리 목록: {cats}
#
# 아래 조건에 따라 답변해줘:
#
# 1. 각 카테고리에 대해, 해당 단어가 **그 카테고리 안에서 스스로 정의 가능한 개념**인지 판단하라.
#    - 단지 관련이 있거나 영향을 주는 것은 정의로 간주하지 않는다.
#    - 예를 들어, "밀"은 "질병"과 관련된 알레르기 원인이 될 수 있지만, "질병"이라는 카테고리 내에서 정의되는 개념이 아니므로 제외해야 한다.
#    - 예를 들어, "밀은" 효율적인 생산을 위해 다양한 '기술'이 적용되지만 밀이 의미하는 것이 '기술'이란 카테고리 내에서 관련히 전혀 없으므로 제외해야 한다.
#
# 2. 각 카테고리에 대해 가능한 경우 이 출력 형식을 절대적으로 따르라(무조건 이렇게 출력해야 한다):
#
#     정의: (정확하고 간결한 정의, 반드시 ~다로 끝나는 문장)
#     예문: (자연스러운 예문, 반드시 ~다로 끝나는 문장)
# 제발 저 두 정의, 예문만 출력해라, 카테고리 이름과 단어 이름 위에 같이 출력하지 마라.
#
# 3. 해당 카테고리에서 정의할 수 없으면 절대 출력하지 마라.
#     그냥 그 카테고리 자체를 출력하지 마라.
#     출력 하지 않는다는 메세지도 출력하지 마라
#
# 4. 만약 {information} 단어가 같은 실체를 가리키거나 여러 카테고리에서 연관이 있다면 그것들을 하나로 묶어 [카테고리1, 카테고리2, ...] 형식으로 하나의 정의와 예문을 작성하라.
#
# 5. 단, 다른 실체를 의미 하는 경우에는 별도로 작성하라.
# 예를 들어 "배"는 과일(생명체, 음식)과 선박(기계, 기술)으로 **완전히 다른 실체**를 의미하므로 나누어야 한다.
#
# 6. 비슷한 단어(예: "배" → "배터리", "배선")가 아니라 정확히 입력된 단어 그대로만 판단해야 한다.
#
# 7. 중복되어 출력되지 않도록 한다.
#
# 반드시 한국어로만 출력하라.
# """
#
# prompt = PromptTemplate(input_variables=["cats", "information"], template=summery_template)
# llm = ChatOpenAI(temperature=1, model_name="gpt-4o")
# chain = prompt | llm | StrOutputParser()
#
# # 4. 엔드포인트 정의
# @app.post("/define")
# async def define_word(req: DefinitionRequest):
#     try:
#         if not req.cats or not req.information:
#             raise HTTPException(status_code=400, detail="단어와 카테고리를 모두 입력하세요.")
#
#         result = chain.invoke(input={
#             "cats": req.cats,
#             "information": req.information.strip()
#         })
#
#         return {"result": result}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")
