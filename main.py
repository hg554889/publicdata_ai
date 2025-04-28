from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# .env 불러오기
load_dotenv()

# 모델과 데이터 로딩
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "response_reordered_verified.jsonl")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

documents = [f"{item['prompt']} {item['response']}" for item in data]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI 인스턴스
app = FastAPI()

# 요청 모델
class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
async def recommend_classes(request: QueryRequest):
    user_query = request.query

    # Query embedding
    query_vec = model.encode([user_query])
    query_vec = np.array(query_vec)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # 유사도 계산
    similarities = np.dot(embeddings, query_vec.T)
    top_k = 5
    I = np.argsort(similarities, axis=0)[::-1][:top_k]
    retrieved_context = [documents[idx] for idx in I.flatten()]

    context = "\n".join(f"- {doc}" for doc in retrieved_context)

    try:
        response = client.chat.completions.create(
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
        return {"answer": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
