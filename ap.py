import os
import streamlit as st
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO  # io 모듈에서 BytesIO 임포트

# OpenAI API 설정
api_key = st.secrets["OPENAI_API_KEY"]

st.title("Show your Food!!")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

system_message = ''' 
너의 이름은 food classifier bot이야.
너는 항상 존댓말을 하는 챗봇이야. 절대로 다나까는 쓰지말고 '요'높임말로 끝내.
너는 음식 사진을 받으면 그 사진 속 음식이 무엇인지 한글로 대답해.
영어로 질문을 받아도 무조건 한글로 답변해줘.
한글이 아닌 답변일 때는 다시 생각해서 한글로 만들어줘
음식이 아닌 사진을 받았을 때는 "음식이 아닙니다. 음식사진을 업로드해주세요" 라고 대답을 해.
'''
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_message})

uploaded_file = st.file_uploader("음식 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

# Function to encode the image to base64
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
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 사진속 음식은 무엇인지 말해주고 사진속 음식의 칼로리는 몇인지 대답해"
                                    "추측의 말투로 말을 하지말고 확신에 찬 말투로 대답해줘. 잘어울리는 음식은 너가 생각하기에 최고로 잘어울리는 음식으로 추천해 .대답을 할 때는"
                                    "음식 :  "
                                    "칼로리: "
                                    "영양성분:  "
                                    "잘어울리는 음식:"
                                    "으로 대답해."
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
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.write(result['choices'][0]['message']['content'])
        else:
            st.error("OpenAI API 호출에 실패했습니다.")
            st.error(f"상태 코드: {response.status_code}, 응답: {response.text}")

    except Exception as e:
        st.error("이미지 처리에 실패했습니다. 다른 이미지를 업로드해 주세요.")
        st.error(f"오류 내용: {e}")

for message in st.session_state.messages:
    if message["role"] != "system":
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
            ]
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.write(result['choices'][0]['message']['content'])
            st.session_state.messages.append({"role": "assistant", "content": result['choices'][0]['message']['content']})
        else:
            st.error("OpenAI API 호출에 실패했습니다.")
            st.error(f"상태 코드: {response.status_code}, 응답: {response.text}")
