"""
Description: 
Author: PeterYoung
Date: 2025-02-07 10:59:38
LastEditTime: 2025-02-07 11:40:22
LastEditors: PeterYoung
"""
import BEN2
import onnx
import onnxruntime
import torch
from PIL import Image
import torchvision

# Assuming the function BEN2.BEN_Base() returns a PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BEN2.BEN_Base().to(device).eval()  # init model

# Load the checkpoint - assuming it's a PyTorch state_dict
model.loadcheckpoints("PramaLLC/BEN2/BEN2_Base.pth")

# Dummy input corresponding to the expected input dimensions of the model
# Here we assume (1, 3, 224, 224) as an example; modify if needed
dummy_input = torch.randn(1, 3, 1024, 1024, device=device)


# Export the model to ONNX
onnx_file_path = "./BEN2_Base.onnx"
torch.onnx.export(model, dummy_input, onnx_file_path, opset_version=15)
print(f"Model exported to {onnx_file_path} successfully")