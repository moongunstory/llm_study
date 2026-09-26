import torch
from torch import nn

# 1. 입력 데이터
# batch_size = 1
# sequence_length = 4
# embedding_dim = 8

x = torch.tensor([
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ]
])

# 2. 셀프-어텐션 모델
class SelfAttention(nn.Module):
    def __init__(self):
        super.__init__()