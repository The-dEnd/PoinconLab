import torch
model = torch.load('yolo/model1/runs/train/exp/weights/best.pt', map_location='cpu')
torch.save(model, 'new_best.pt')
