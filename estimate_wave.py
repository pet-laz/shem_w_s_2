#!/usr/bin/env python3
# estimate_wave.py
import cv2, numpy as np, pathlib, csv, math
from scipy.optimize import curve_fit

SRC_DIR   = pathlib.Path("captchas_preprocessed_only")
N_IMAGES  = 100                 # сколько первых файлов анализировать
OUT_CSV   = "wave_params.csv"   # построчный лог

def sin_model(x, A, phi, b, P):
    return b + A*np.sin(2*np.pi*x/P + phi)

def get_midline(mask):
    h, w = mask.shape
    xs = np.arange(w)
    ys = np.array([np.mean(np.where(mask[:,x])[0]) if mask[:,x].any() else math.nan
                   for x in xs], dtype=np.float32)
    ys = np.interp(xs, xs[~np.isnan(ys)], ys[~np.isnan(ys)])
    return xs, ys

def analyse(img_path, P_guess):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    xs, ys = get_midline(bw)
    A0 = (ys.max()-ys.min())/2; b0 = ys.mean(); phi0 = 0
    (A, phi, b), _ = curve_fit(
        lambda x, A, phi, b: sin_model(x, A, phi, b, P_guess),
        xs, ys, p0=[A0, phi0, b0], maxfev=10000)
    return A, P_guess

def main():
    files = sorted(SRC_DIR.glob("*.jp*g"))[:N_IMAGES]
    assert files, "Нет файлов в папке!"
    # грубая оценка P – половина ширины капчи
    sample = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    P0 = sample.shape[1] / 1.2
    params = []
    for p in files:
        A, P = analyse(p, P0)
        params.append((p.name, round(A,2), round(P,2)))
        print(f"{p.name:25s}  A={A:5.2f}px  P≈{P:.1f}px")
    # сохраняем CSV
    with open(OUT_CSV, "w", newline='', encoding="utf8") as f:
        csv.writer(f).writerows([("file","A_px","P_px")]+params)
    A_mean = np.mean([r[1] for r in params])
    P_mean = np.mean([r[2] for r in params])
    print(f"\nСредняя амплитуда  Ā = {A_mean:.2f}px")
    print(f"Средний период     P̄ = {P_mean:.2f}px")

if __name__ == "__main__":
    main()
