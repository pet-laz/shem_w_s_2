#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deskew_captchas.py
------------------
• Берёт все *.png / *.jpg / *.jpeg из каталога SRC_DIR
• Делит изображение на вертикальные «окна» шириной WINDOW пикселей
• В каждом окне ищет верхнюю и нижнюю точку белого контура
  -> середина = локальный baseline
• Сглаживает полученную ломаную (GaussianBlur)
• Сдвигает каждый столбец так, чтобы baseline совпал с глобальным средним
• Сохраняет результат в DST_DIR под тем же именем

Параметры (WINDOW, SMOOTH_K, SCALE…) правятся в блоке CONFIG.
"""

# ---------- CONFIG ----------
SRC_DIR   = "captchas_preprocessed_only"     # входная папка
DST_DIR   = "captchas_deskewed"              # куда писать результат
WINDOW    = 3        # ширина «окна» (N) — шаг дискретизации волны, px
SMOOTH_K  = 21       # сглаживание центров (нечётное, GaussianBlur ksize)
BIN_INV   = False     # True, если буквы белые на чёрном фоне
SCALE     = 1.0      # множитель смещения (1.0 = «как есть», >1 усилит эффект)

SAVE_DEBUG_PLOT = False   # True => сохраняет PNG с графиком центров
# ----------------------------

import cv2
import numpy as np
import pathlib
from datetime import datetime

SRC = pathlib.Path(SRC_DIR)
DST = pathlib.Path(DST_DIR)
DST.mkdir(exist_ok=True)

def column_centers(mask: np.ndarray, win: int):
    """Возвращает xs, ys — координаты центров маски в каждом 'окне'."""
    h, w = mask.shape
    xs = np.arange(0, w, win)
    ys = np.empty_like(xs, dtype=np.float32)

    for i, x0 in enumerate(xs):
        col = mask[:, x0:x0 + win]
        ys_in_col = np.where(col)[0]        # строки, где mask == 1
        if ys_in_col.size:
            ys[i] = (ys_in_col.min() + ys_in_col.max()) / 2.0
        else:                               # пустой столбец → NaN
            ys[i] = np.nan

    # интерполируем пропуски NaN линейно
    nan_mask = np.isnan(ys)
    if nan_mask.any():
        ys[nan_mask] = np.interp(xs[nan_mask], xs[~nan_mask], ys[~nan_mask])
    return xs.astype(np.float32), ys

def smooth(y: np.ndarray, k: int):
    """GaussianBlur вдоль оси X (k должен быть нечётным)."""
    k = max(3, k | 1)              # нечётное
    return cv2.GaussianBlur(y.reshape(1, -1), (k, 1), 0).flatten()

def deskew(gray: np.ndarray, xs, centers, baseline, scale=1.0, flip=False):
    h, w = gray.shape
    shift = (baseline - np.interp(np.arange(w), xs, centers)    # baseline - center
             if flip else
             np.interp(np.arange(w), xs, centers) - baseline)   # center - baseline
    shift *= scale

    map_y = (np.arange(h)[:, None] - shift).astype(np.float32)
    map_x = np.repeat(np.arange(w)[None, :], h, axis=0).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

def process_image(path: pathlib.Path):
    name = path.name
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    # --- бинаризация ---
    th_flag = cv2.THRESH_BINARY_INV if BIN_INV else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, th_flag + cv2.THRESH_OTSU)

    # небольшая морфология → убираем шум
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    xs, ys = column_centers(bw, WINDOW)
    ys_s   = smooth(ys, SMOOTH_K)
    baseline = float(np.mean(ys_s))

    corrected = deskew(gray, xs, ys_s, baseline, SCALE, True)
    cv2.imwrite(str(DST / name), corrected)

    # ----- отладочный график -----
    if SAVE_DEBUG_PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.imshow(gray, cmap='gray')
        plt.plot(xs, ys_s, 'r-', linewidth=1)
        plt.title(name)
        plt.axis('off')
        plt.tight_layout()
        debug_name = DST / f"{path.stem}_debug.png"
        plt.savefig(debug_name, dpi=120)
        plt.close()

    print(f"{name:30s}  shift range ±{np.ptp(ys_s - baseline):4.1f} px")

def main():
    images = [p for p in SRC.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    if not images:
        print("❌  В папке нет подходящих изображений")
        return

    print(f"Start  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Images: {len(images)}  |  window={WINDOW}px  smooth={SMOOTH_K}  scale={SCALE}")
    for p in images:
        process_image(p)

    print("✓ Готово! Выровненные капчи лежат в", DST.resolve())

if __name__ == "__main__":
    main()
