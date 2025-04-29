from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import openai  # OpenAI 직접 호출
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 불러오기
load_dotenv()

# FastAPI 인스턴스
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델과 데이터 로딩 (서버 시작 시 한 번만 실행)
logger.info("모델과 데이터 로딩 시작...")
start_time = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "response_reordered_verified_replaced.jsonl")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

documents = [f"{item['prompt']} {item['response']}" for item in data]

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 모델 로드 및 GPU로 이동
model = SentenceTransformer("all-mpnet-base-v2")
model.to(device)

# 임베딩 미리 계산
logger.info("문서 임베딩 계산 중...")
embeddings = model.encode(documents, show_progress_bar=True, device=device)
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# OpenAI 클라이언트 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

logger.info(f"모델 및 데이터 로딩 완료: {time.time() - start_time:.2f}초 소요")

# 요청 모델
class QueryRequest(BaseModel):
    query: str

# 상태 확인 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# 쿼리 인코딩 (동기 처리)
def encode_query(query_text):
    return model.encode([query_text], device=device)

@app.post("/recommend")
async def recommend_classes(request: QueryRequest):
    start_time = time.time()
    user_query = request.query
    logger.info(f"요청 받음: {user_query}")

    try:
        # Query embedding (동기로 처리)
        query_vec_raw = encode_query(user_query)
        query_vec = np.array(query_vec_raw)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        logger.info(f"쿼리 인코딩 완료: {time.time() - start_time:.2f}초 소요")

        # 유사도 계산
        similarities = np.dot(embeddings, query_vec.T)
        top_k = 5
        I = np.argsort(similarities, axis=0)[::-1][:top_k]
        retrieved_context = [documents[idx] for idx in I.flatten()]

        context = "\n".join(f"- {doc}" for doc in retrieved_context)

        logger.info(f"유사 문서 검색 완료: {time.time() - start_time:.2f}초 소요")

        # OpenAI API 호출 (동기 방식으로 수정)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""당신은 대학 강의 추천 챗봇입니다.
다음 정보를 참고하여 사용자의 질문에 맞는 학과를 3개 추천해 주세요. 
1.만약 학과 수업이 없다면, 질문 받은 학과와 관련된 수업을 3개 추천해 주세요. 
2.비서같이 안내하는 문장으로 이야기 하여 주세요.
3.사용자에게 다시 질문하는 문장은 사용하지 마세요.

[참고 정보]
{context}

[질문]
{user_query}

[답변]
"""
            }],
            temperature=0.2
        )

        logger.info(f"총 처리 시간: {time.time() - start_time:.2f}초")
        return {"answer": response['choices'][0]['message']['content']}

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 미들웨어 추가 - 요청 처리 시간 로깅
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"요청 처리 시간: {process_time:.2f}초")
    return response
