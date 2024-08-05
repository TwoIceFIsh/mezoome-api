import joblib
import torch
import torch.nn as nn
from pydantic import BaseModel


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
def predict(request: PredictRequest):
    try:
        new_X = vectorizer.transform(request.titles).toarray()
        new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(new_X_tensor).cpu().numpy().flatten()
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return None