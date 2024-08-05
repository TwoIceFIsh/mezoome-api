import logging

import joblib
import requests
import torch
import torch.nn as nn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pymongo import MongoClient

from naver.naver_articles import get_naver_articles

# FastAPI 인스턴스 생성
app = FastAPI()


# 모델 정의
class FearGreedModel(nn.Module):
    def __init__(self, input_dim):
        super(FearGreedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 모델 및 CountVectorizer 로드
def initialize_model(input_dim, model_path):
    model = FearGreedModel(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드로 전환
    return model


# 저장된 CountVectorizer 로드
vectorizer = joblib.load('vectorizer.pkl')

# 모델 불러오기
model = initialize_model(input_dim=len(vectorizer.get_feature_names_out()), model_path='fear_greed_model_gpu.pth')


# 요청 모델 정의
class PredictRequest(BaseModel):
    titles: list


# FastAPI 엔드포인트 정의
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        new_X = vectorizer.transform(request.titles).toarray()
        new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(new_X_tensor).cpu().numpy().flatten()
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 모델 평가 엔드포인트 정의
@app.get("/update_index")
def run_update_index():
    return update_index()


def update_index():
    data = {}
    # 새로운 게시물을 업데이트 한다.
    if get_naver_articles():
        # MongoDB에 연결합니다.
        client = MongoClient(
            host='',
            port=27017,
            username='',
            password='',
            authSource='mezoome'
        )
        db = client['mezoome']
        collection = db['mezoome_articles']

        # prediction 필드가 없는 문서들을 찾습니다.
        articles = collection.find({"prediction": {"$exists": False}})

        # data를 리스트로 변환합니다.

        # data를 가져와서 하나씩 predict를 하고 데이터를 업데이트한다.
        for article in articles:
            new_X = vectorizer.transform([article['subject']]).toarray()
            new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(device)
            with torch.no_grad():
                prediction = model(new_X_tensor).cpu().numpy().flatten()
            prediction_value = float(prediction[0])
            collection.update_one({"_id": article['_id']}, {"$set": {"prediction": prediction_value}})

        # MongoDB 연결을 닫습니다.
        client.close()

        # 인덱스 생성 요청을 보냅니다.
        res2 = requests.get("https://mezoo.me/api/v1/make-mzm-index")
        print(res2.json())
        res3 = requests.get("https://mezoo.me/api/v1/make-fag-index")
        print(res3.json())

        data = res2.json()

        return {"status": data}


def scheduled_task():
    logging.info("Scheduled task executed")
    if update_index():
        logging.info("Jobs with status 'DONE' processed.")
    else:
        pass


# APScheduler를 사용하여 스케줄링 작업 설정
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_task, 'cron', minute=0, id='scheduled_task')
scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()