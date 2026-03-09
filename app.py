import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
from basicsr.archs.rrdbnet_arch import RRDBNet

# -----------------------------
# Load ESRGAN Model
# -----------------------------
model_path = "RealESRGAN_x4plus.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
loadnet = torch.load(model_path, map_location=device)

if "params_ema" in loadnet:
    loadnet = loadnet["params_ema"]

model.load_state_dict(loadnet, strict=True)
model.eval()
model = model.to(device)

print("Model loaded successfully")

# -----------------------------
# Variables
# -----------------------------
input_image = None
enhanced_image = None

# -----------------------------
# Upload Image
# -----------------------------
def upload_image():
    global input_image

    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

    if not path:
        return

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_image = img

    display_image(img)
    status_label.config(text="Image uploaded")

# -----------------------------
# Enhance Image
# -----------------------------
def enhance_image():
    global input_image, enhanced_image

    if input_image is None:
        messagebox.showerror("Error", "Please upload an image first")
        return

    status_label.config(text="Enhancing image... please wait")
    root.update()

    img = input_image.astype(np.float32) / 255.0
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().cpu().clamp_(0,1).numpy()
    output = np.transpose(output,(1,2,0))
    output = (output*255).astype(np.uint8)

    enhanced_image = output

    display_image(output)

    status_label.config(text="Enhancement complete")

# -----------------------------
# Save Image
# -----------------------------
def save_image():
    global enhanced_image

    if enhanced_image is None:
        messagebox.showerror("Error","No enhanced image to save")
        return

    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG File","*.png"),("JPG File","*.jpg")]
    )

    if path:
        cv2.imwrite(path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
        status_label.config(text="Image saved successfully")

# -----------------------------
# Display Image
# -----------------------------
def display_image(img):
    img = Image.fromarray(img)

    img.thumbnail((500,500))

    imgtk = ImageTk.PhotoImage(img)

    image_panel.config(image=imgtk)
    image_panel.image = imgtk

# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()
root.title("ESRGAN Image Enhancer")

# Wider window
root.geometry("900x650")

title = tk.Label(root,text="AI Image Super Resolution (ESRGAN)",font=("Arial",18,"bold"))
title.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame,text="UPLOAD IMAGE",width=20,height=2,command=upload_image)
upload_btn.grid(row=0,column=0,padx=10)

enhance_btn = tk.Button(btn_frame,text="ENHANCE IMAGE",width=20,height=2,command=enhance_image)
enhance_btn.grid(row=0,column=1,padx=10)

save_btn = tk.Button(btn_frame,text="SAVE IMAGE",width=20,height=2,command=save_image)
save_btn.grid(row=0,column=2,padx=10)

image_panel = tk.Label(root)
image_panel.pack(pady=20)

status_label = tk.Label(root,text="Upload an image to begin",font=("Arial",12))
status_label.pack(pady=10)

root.mainloop()