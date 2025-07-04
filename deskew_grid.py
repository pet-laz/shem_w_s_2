#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deskew_grid.py
--------------
Использование:
    python deskew_grid.py path/to/captcha.jpg --out deskew_variants

• На входе — одна картинка.
• Для каждой комбинации WINDOW × SMOOTH_K × SCALE делает выравнивание
  и сохраняет файл "<stem>_W<W>_K<K>_S<S>_F<f>.png"
  где F=0/1 — применён ли flip_shift.
"""

import cv2
import numpy as np
import pathlib, argparse, itertools, sys

# ---------- Наборы параметров ----------
WINDOWS  = [2, 4, 6]           # px
SMOOTHS  = [9, 15, 25]         # px, нечётное
SCALES   = [1.0, 2.0, 3.0]     # множитель сдвига
FLIP_SHIFT = True             # True -> смещение поменяет знак
BIN_INV    = False             # False, если буквы белые
# ----------------------------------------

# ---------- базовые функции -------------
def column_centers(mask: np.ndarray, win: int):
    h, w = mask.shape
    xs = np.arange(0, w, win)
    ys = np.empty_like(xs, dtype=np.float32)

    for i, x0 in enumerate(xs):
        col = mask[:, x0:x0+win]
        ys_in = np.where(col)[0]
        ys[i] = (ys_in.min() + ys_in.max()) / 2.0 if ys_in.size else np.nan

    # линейная интерполяция по NaN
    if np.isnan(ys).any():
        ys = np.interp(xs, xs[~np.isnan(ys)], ys[~np.isnan(ys)])
    return xs.astype(np.float32), ys

def smooth(y: np.ndarray, k: int):
    k = max(3, k | 1)               # нечётное
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
# ----------------------------------------

def process_one(src_path: pathlib.Path, dst_dir: pathlib.Path):
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        print("⚠️  Не удалось прочитать", src_path)
        return
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- бинаризация ---
    th_flag = cv2.THRESH_BINARY_INV if BIN_INV else cv2.THRESH_BINARY
    _, bw = cv2.threshold(gray, 0, 255, th_flag + cv2.THRESH_OTSU)

    # небольшой open, чтобы убрать шум
    bw = cv2.morphologyEx(
            bw, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)

    stem = src_path.stem
    for W, K, S in itertools.product(WINDOWS, SMOOTHS, SCALES):
        xs, ys    = column_centers(bw, W)
        ys_s      = smooth(ys, K)
        baseline  = float(np.mean(ys_s))
        corrected = deskew(gray, xs, ys_s, baseline, S, FLIP_SHIFT)

        out_name = f"{stem}_W{W}_K{K}_S{S}_F{int(FLIP_SHIFT)}.png"
        cv2.imwrite(str(dst_dir / out_name), corrected)
        print(f"→ {out_name:25s} | shift ±{np.ptp(ys_s-baseline)*S:.1f}px")

def main():
    parser = argparse.ArgumentParser(description="Deskew captcha with param grid")
    parser.add_argument("image", type=pathlib.Path, help="captchas_preprocessed_only/captcha_1150_kPnHbtTy.jpeg")
    parser.add_argument("--out", type=pathlib.Path, default="deskew_variants",
                        help="captchas_d")
    args = parser.parse_args()

    if not args.image.exists():
        print("❌  Файл не найден:", args.image)
        sys.exit(1)

    args.out.mkdir(exist_ok=True, parents=True)
    process_one(args.image, args.out)
    print("✓ Готово! Все варианты →", args.out.resolve())

if __name__ == "__main__":
    main()
