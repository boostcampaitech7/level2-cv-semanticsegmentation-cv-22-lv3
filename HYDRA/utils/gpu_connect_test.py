import torch
print(torch.cuda.is_available())  # True 출력
print(torch.cuda.get_device_name(0))  # GPU 이름 출력

if torch.cuda.is_available():
    print("사용 가능한 GPU 이름:", torch.cuda.get_device_name(0))
    print("총 GPU 메모리 (MB):", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    print("현재 사용 중인 메모리 (MB):", torch.cuda.memory_allocated(0) / 1024 / 1024)
    print("현재 사용 가능한 메모리 (MB):", torch.cuda.memory_reserved(0) / 1024 / 1024)
else:
    print("사용 가능한 GPU가 없습니다.")