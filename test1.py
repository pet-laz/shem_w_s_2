import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob, pathlib

TARGET = cv2.imread("orig.png", cv2.IMREAD_GRAYSCALE)   # ваша капча
_, TARGET = cv2.threshold(TARGET, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

word = "kpnHbtTy"
fonts = glob.glob("C:/Windows/Fonts/*.ttf")

def render(ttf):
    font = ImageFont.truetype(ttf, 40)
    img = Image.new("L", (TARGET.shape[1], TARGET.shape[0]), 0)
    d   = ImageDraw.Draw(img)
    d.text((5,5), word, 255, font=font)
    arr = np.array(img)
    _, arr = cv2.threshold(arr, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return arr

def score(img):
    # простая метрика: Jaccard (IoU) масок
    inter = np.logical_and(img, TARGET).sum()
    union = np.logical_or (img, TARGET).sum()
    return inter / union if union else 0

best = sorted([(score(render(f)), pathlib.Path(f).stem) for f in fonts],
              reverse=True)[:10]
for s,name in best:
    print(f"{name:15s}  {s*100:.1f}% совпадения")
