import torch

# MPS 장치가 사용 가능한지 확인합니다.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 장치가 사용됩니다.")
else:
    device = torch.device("cpu")
    print("MPS 장치가 사용되지 않으며, CPU가 사용됩니다.")
