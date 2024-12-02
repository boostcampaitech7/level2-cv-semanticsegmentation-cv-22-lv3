# test_model_initialization.py

from models import initialize_model

try:
    model = initialize_model(num_classes=28, pretrained=True)
    print("모델 초기화 성공")
except Exception as e:
    print(f"모델 초기화 중 오류 발생: {e}")
