import os
import streamlit as st
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Food Analysis",
    page_icon="🍽",
    layout="centered",
)
# Streamlit 제목 설정
st.title("Show your Food!!")



# 초기 안내 메시지 추가
st.markdown("""
            ### <이용방법>
            이 앱은 업로드한 음식 사진을 분석하여 음식의 이름과 칼로리, 영양 성분을 알려드립니다.

            1. 비밀번호를 입력하세요.
            2. 음식 사진을 업로드하세요.
            3. 챗봇이 자동으로 분석을 수행하고 결과를 알려드릴 것입니다.

            ‼️주의‼️: 음식 사진만 업로드해주세요. 다른 사진은 인식되지 않을 수 있습니다.
    <style>
        .reportview-container {
            background-color: #f3f4ed;
        }
        h1 {
            color: black;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

stored_password = st.secrets["password"]
api_key = st.secrets["OPENAI_API_KEY"]

input_password = st.text_input("비밀번호를 입력하세요", type="password")

if input_password:
    if input_password == stored_password:
        st.success("비밀번호가 확인되었습니다. API Key가 자동으로 입력됩니다.")
    else:
        st.error("비밀번호가 올바르지 않습니다.")
        api_key = None
else:
    api_key = None

if api_key:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"
    
    # Few-shot 예시 메시지 설정
    few_shot_examples = [
         {"role": "assistant", "content": "음식 분석:\n"
                                     "1. 음식의 종류를 확인합니다.\n"
                                     "2. 해당 음식의 칼로리를 추정합니다.\n"
                                     "3. 음식의 영양성분(탄수화물, 단백질, 지방)을 계산합니다.\n"
                                     "4. 이 음식과 잘 어울리는 다른 음식을 추천합니다.\n"
                                     "\n결과:\n"
                                     "- 음식: 비빔밥\n"
                                     "- 칼로리: 550kcal\n"
                                     "- 영양성분: 탄수화물 80g, 단백질 12g, 지방 10g\n"
                                     "- 잘 어울리는 음식: 샐러드"},
         {"role": "assistant", "content": "음식 분석:\n"
                                     "1. 음식의 종류를 확인합니다.\n"
                                     "2. 해당 음식의 칼로리를 추정합니다.\n"
                                     "3. 음식의 영양성분(탄수화물, 단백질, 지방)을 계산합니다.\n"
                                     "4. 이 음식과 잘 어울리는 다른 음식을 추천합니다.\n"
                                     "\n결과:\n"
                                     "- 음식: 김밥\n"
                                     "- 칼로리: 300kcal\n"
                                     "- 영양성분: 탄수화물 50g, 단백질 8g, 지방 7g\n"
                                     "- 잘 어울리는 음식: 닭가슴살"}
    ]
    
    system_message = ''' 
    너의 이름은 food classifier bot이야.
    너는 항상 한국어로 대답하고, 존댓말을 해.
    음식을 분석할 때는 다음 단계를 따르며 체계적으로 분석해.
    1. 사진 속의 음식이 무엇인지 확인해.
    2. 음식의 이름을 바탕으로 칼로리를 추정해.
    3. 칼로리를 추정한 후에 영양분(탄수화물, 단백질, 지방 등)의 양을 각각 추정해.
    4. 이 음식에 부족한 영양분을 채워줄 수 있는 다른 음식을 제안해.
    음식이 아닌 사진을 받았을 때는 "음식이 아닙니다. 음식 사진을 업로드해주세요" 라고 대답을 해.
    '''
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "system", "content": system_message})
        st.session_state.messages.extend(few_shot_examples)  # Few-shot 예시 추가

    uploaded_file = st.file_uploader("음식 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

    # 이미지 base64로 인코딩하는 함수
    def encode_image(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    if uploaded_file is not None:
        # 이미지 열기 및 처리
        image = Image.open(uploaded_file)
        image = image.convert("RGB")  # 이미지를 RGB로 변환

        st.image(image, caption="업로드한 이미지", use_column_width=True)

        try:
            # 이미지를 base64로 인코딩
            base64_image = encode_image(image)

            # OpenAI API 요청 준비
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": st.session_state["openai_model"],
                "messages": st.session_state.messages + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "이 사진 속 음식이 무엇인지 알아보고, 사진 속 음식의 칼로리를 계산해줘. 추측하지 말고 확신을 가지고 대답해줘."
                                       "분석할 때는 다음 단계를 따라줘:\n"
                                        "1. 이 사진 속 음식이 무엇인지 파악해.\n"
                                        "2. 음식의 이름을 바탕으로 칼로리를 계산해.\n"
                                        "3. 음식의 영양성분을 추정해 (탄수화물, 단백질, 지방 등).\n"
                                        "4. 이 음식의 부족한 영양성분을 채워주거나, 같이 먹으면 조합이 좋은 음식을 추천해.\n"
                                        "답변은 다음과 같은 형식으로 줘:\n"
                                        "- 음식: [음식 이름]\n"
                                        "- 칼로리: [칼로리 정보]kcal\n"
                                        "- 영양성분: [탄수화물 정보]g, [단백질 정보]g, [지방 정보]g\n"
                                        "- 잘 어울리는 음식: [추천 음식]"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.5  # 정밀도를 높이고 창의성을 줄이기 위한 temperature 설정
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.write(result['choices'][0]['message']['content'])


                # 추가 질문 옵션 추가
                st.markdown("### 추가 질문이 있나요?")
                additional_question = st.text_input("추가 질문을 입력하세요:")

                if st.button("추가 질문 보내기"):
                    if uploaded_file is not None:
                        base64_image = encode_image(image)  # 이미지를 base64로 인코딩
                        
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": additional_question
                        })
                        
                        st.write("질문이 제출되었습니다. 잠시만 기다려주세요...")



                    

                        # 추가 질문을 OpenAI API로 보내고 응답 처리
                        payload = {
                            "model": st.session_state["openai_model"],
                            "messages": st.session_state.messages + [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": additional_question
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            "temperature": 0.2  # 정밀도를 높이고 창의성을 줄이기 위한 temperature 설정
                        }

                        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

                        if response.status_code == 200:
                            result = response.json()
                            st.write(result['choices'][0]['message']['content'])
                            st.session_state.messages.append({"role": "assistant", "content": result['choices'][0]['message']['content']})
                        else:
                            st.error("추가 질문에 대한 OpenAI API 호출에 실패했습니다.")
                            st.error(f"상태 코드: {response.status_code}, 응답: {response.text}")
                    else:
                        st.warning("이미지 없이 추가 질문을 보낼 수 없습니다. 먼저 이미지를 업로드하세요.")
                # 피드백 기능 추가
                st.markdown("### 이 응답이 도움이 되었나요?")
                feedback = st.radio("응답에 대한 피드백을 선택하세요", ["도움이 되었어요", "더 개선이 필요해요"], index=None)

                if feedback == "더 개선이 필요해요":
                    feedback_text = st.text_area("어떤 부분을 개선하면 좋을지 알려주세요:")
                    if st.button("피드백 제출"):
                        # 피드백을 파일에 저장
                        with open("feedback.txt", "a") as f:
                            f.write(f"시간: {datetime.now()}\n")
                            f.write(f"사용자 피드백: {feedback_text}\n")
                            f.write("-" * 40 + "\n")
                        st.success("피드백이 제출되었습니다. 감사합니다!")
                    else:
                        st.warning("피드백 내용을 입력해주세요.")
                elif feedback == "도움이 되었어요":
                    st.success("""감사합니다! 식사 맛있게하세요🍽️\n
                               This program was created by La-sinhyeok & Park geonsoo""")
                    with open("feedback.txt", "a") as f:
                            f.write(f"시간: {datetime.now()}\n")
                            f.write(f"사용자 피드백: {"good"}\n")
                            f.write("-" * 40 + "\n")
            else:
                st.error("OpenAI API 호출에 실패했습니다.")
                st.error(f"상태 코드: {response.status_code}, 응답: {response.text}")

        except Exception as e:
            st.error("이미지 처리에 실패했습니다. 다른 이미지를 업로드해 주세요.")
            st.error(f"오류 내용: {e}")

    for message in st.session_state.messages:
        if message["role"] not in ["system", "assistant", "user"]:  # 사용자 메시지만 표시
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("채팅을 계속하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # OpenAI API 호출 (기본 텍스트만 처리)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": st.session_state["openai_model"],
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                "temperature": 0.2  # 정밀도를 높이고 창의성을 줄이기 위한 temperature 설정
            }

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.write(result['choices'][0]['message']['content'])
                st.session_state.messages.append({"role": "assistant", "content": result['choices'][0]['message']['content']})
            else:
                st.error("OpenAI API 호출에 실패했습니다.")
                st.error(f"상태 코드: {response.status_code}, 응답: {response.text}")
else:
    st.warning("API Key를 입력하세요.")
