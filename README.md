Food Classifier
=================


## Motivation & purpose
> 내가 먹는 음식의 칼로리 및 영양성분은 어떨까? 식단 정보를 쉽게 알 수 있지 않을까?
> 내가 먹으려는 음식의 영양성분을 사진 찍어 쉽게 정보를 얻자, 나에게 맞는 식단 정보를 쉽게 얻자 


Project Introdution
----------------

음식 사진을 업로드 하면 음식에 관한 칼로리와 영양정보를 알려주는 챗봇 제작. 
Streamlit 기반 UI.


Development Environment
---------------


▶모델 제작

-VisualStudioCode

-Python

-OpenAI


▶UI구현

-Streamlit


Key features
-------------


- OpenAI_API_Key(password) 입력

- 이미지 업로드

- GPT-4o 모델로부터 응답 생성

- 응답에 대한 추가 질문 or 피드백


## Future Work
------------

- few-shot prompting을 할 때 지금의 경우 단지 텍스트 정보만 들어있는데 이미지와 함께 원하고자 하는 프롬프팅을 할 수 있다면 모델의 성능을 더 높일 수 있을 것이다.

- 현재 모델의 성능에 대한 정확도나 만족도에 대한 평가가 코드를 돌린 컴퓨터에만 텍스트 파일로 저장이 되는데 이를 개선시켜 사용자들의 앱 이용 평가를 받아 모을 수 있는 서버를 구축하여 데이터베이스를 얻을 수 있다면 모델 성능을 더 개선할 수 있을 것이다.


Link
-----------
https://foodloris-classify-meuhscmqrzch5dqk4xwbrh.streamlit.app
