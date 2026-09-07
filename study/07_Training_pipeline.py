import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

x = torch.randn(1000, 2)

y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)

# 전체 데이터 1000개
# Train 700개
# Validation 150개
# Test 150개
# 1. 학습 2. 파라미터 수정 3. 테스트