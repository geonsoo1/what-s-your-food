import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import openai

st.title("HealthCareBot")

# OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

system_message = ''' 
너의 이름은 food classifier bot이야.
너는 항상 존댓말을 하는 챗봇이야. 절대로 다나까는 쓰지말고 '요'높임말로 끝내.
항상 친근하게 대답해줘
너는 음식 사진을 받으면 그 사진 속 음식이 무엇인지 한글로 대답해. 그 음식의 양을 파악하고 칼로리가 몇인지 대답해.
영어로 질문을 받아도 무조건 한글로 답변해줘.
한글이 아닌 답변일 때는 다시 생각해서 한글로 만들어줘
'''

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_message})

# CLIP 모델 및 프로세서 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

uploaded_file = st.file_uploader("음식 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    try:
        # 이미지를 CLIP 모델에 전달하여 텍스트 설명 생성
        inputs = clip_processor(images=image, return_tensors="pt")
        outputs = clip_model.get_text_features(**inputs)

        # 텍스트 생성 (CLIP은 텍스트 생성 기능이 없으므로, 여기에 임의의 설명이 필요)
        # CLIP의 출력은 텍스트 설명을 위한 임베딩이므로, 이를 GPT로 전달하여 설명을 생성
        # 아래는 예시적인 텍스트임을 주의하세요.
        description = "이 이미지는 음식을 나타냅니다."

        # OpenAI GPT 모델과 대화
        prompt = f"이 음식이미지의 음식은?: {description}. 이 음식의 양을 고려했을 때 평균 칼로리는 몇인가요?"

        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        assistant_message = response.choices[0].message['content']
        st.write(assistant_message)

    except Exception as e:
        st.error("이미지 인식에 실패했습니다. 다른 이미지를 업로드해 주세요.")
        st.error(f"오류 내용: {e}")

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("keep chat going..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        )
        assistant_message = response.choices[0].message['content']
        st.markdown(assistant_message)
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
