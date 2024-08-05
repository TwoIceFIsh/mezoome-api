import joblib
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# 1. 데이터 준비
def fetch_data(page):
    url = f"http://localhost:3000/api/v1/parse/{page}"
    response = requests.get(url)
    result = response.json()
    return pd.DataFrame(result['result'])


def load_data(pages):
    all_data = pd.DataFrame()
    for page in pages:
        df = fetch_data(page)
        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data


# 페이지 범위 설정
pages = range(1, 996)  # 1부터 995까지 페이지
df = load_data(pages)

# 2. 데이터 전처리
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['title']).toarray()
y = df['fear_greed_index'].values

# CountVectorizer 저장
joblib.dump(vectorizer, 'vectorizer.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class FearGreedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


train_dataset = FearGreedDataset(X_train, y_train)
test_dataset = FearGreedDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 3. 모델 정의
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


# 모델 초기화
def initialize_model(model_path=None):
    input_dim = X_train.shape[1]  # 현재 데이터의 입력 차원
    model = FearGreedModel(input_dim).to(device)  # 모델을 디바이스로 이동
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))  # map_location을 사용하여 디바이스 설정
            print('Model loaded from', model_path)
        except RuntimeError as e:
            print('Error loading model:', e)
            # 모델 로드 실패 시 초기화
            model = FearGreedModel(input_dim).to(device)
    return model


# 신규학습
# model = initialize_model()
# 재학습
model = initialize_model('fear_greed_model_gpu.pth')


# 4. 모델 학습
def train_model():
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss2 = 0
    epoch = 0
    while True:  # Infinite loop
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 배치를 디바이스로 이동
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        epoch += 1
        if (loss.item() < 2.0):
            print('Early stopping')
            loss2 += 1
            if (loss2 == 3):
                break


train_model()  # 필요한 에폭 수를 설정합니다.

# 모델 저장
torch.save(model.state_dict(), 'fear_greed_model_gpu.pth')
print('Model saved.')

# 5. 예측
model.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 배치를 디바이스로 이동
        outputs = model(X_batch)
        loss = nn.MSELoss()(outputs.squeeze(), y_batch)  # criterion을 여기서 직접 정의
        test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_loader)}')

# 새로운 제목에 대한 예측
new_titles = ['폭락장 이네요']
new_X = vectorizer.transform(new_titles).toarray()
new_X_tensor = torch.tensor(new_X, dtype=torch.float32).to(device)  # 입력 텐서를 디바이스로 이동

model.eval()
with torch.no_grad():
    prediction = model(new_X_tensor).item()
    print(f'Predicted Fear-Greed Index: {prediction}')