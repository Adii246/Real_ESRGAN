import torch
import cv2
import numpy as np
from rrdbnet_arch import RRDBNet

model_path = "RRDB_ESRGAN_x4.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RRDBNet(3, 3, 64, 23, gc=32)

model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

model.eval()
model = model.to(device)

img = cv2.imread("low_res.jpg")

if img is None:
    print("Image not found")
    exit()

img = img.astype(np.float32) / 255.0

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.transpose(img, (2, 0, 1))

img = torch.from_numpy(img).float().unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)

output = output.squeeze().cpu().numpy()

output = np.transpose(output, (1, 2, 0))

output = np.clip(output, 0, 1)

output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

output = (output * 255).astype(np.uint8)

cv2.imwrite("super_resolved.png", output)

print("Done!")