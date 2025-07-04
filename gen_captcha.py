#!/usr/bin/env python3
import re, numpy as np, random, math, pathlib, cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIG -----------------
FONT_TTF      = "fonts/UbuntuMono-R-hinting.ttf"
FONT_SIZE     = 40
STROKE_W      = 1
ALPHABET      = "ABCDEFGHJKMNOPQRSTUWYZabcdefghjkmnopqrstuwyz"
CAPTCHA_LEN   = 8

IMG_H, IMG_W  = 128//2, 300//2#189           # итоговый размер
RENDER_SCALE  = 4                 # мультипликатор разрешения

PADDING       = 40                 # <- число пикселей пустых полей со всех сторон
# волны
A, P          = 14, 60
LOCAL_A, LOCAL_P = 3, 30
SPACING_RANGE = (-6.0, -1.0)
SCALE_Y_RANGE = (1.0, 1.3)
# -----------------------------------------

OUT_DIR  = pathlib.Path("gen_wave")
N_IMAGES = 50000
# random.seed(42); np.random.seed(42)
OUT_DIR.mkdir(exist_ok=True, parents=True)
font = ImageFont.truetype(FONT_TTF, FONT_SIZE * RENDER_SCALE)

def rand_angle(i):
    ang = float(np.random.normal(3, 2))
    if 0 < i < CAPTCHA_LEN-1 and random.random() < 0.1:
        ang = random.choice([-30, 30])
    return ang

def render_flat(text, pad_h):
    """Рисуем ровную строку на холсте high-res."""
    big_w = IMG_W * RENDER_SCALE * 2
    big_h = IMG_H * RENDER_SCALE + 2 * pad_h
    canvas = Image.new("L", (big_w, big_h), 0)
    x = 8 * RENDER_SCALE
    boxes = []
    for i, ch in enumerate(text):
        glyph = Image.new("L", (80*RENDER_SCALE, 80*RENDER_SCALE), 0)
        d = ImageDraw.Draw(glyph)
        d.text((0,0), ch, 255, font=font,
               stroke_width=STROKE_W*RENDER_SCALE,
               stroke_fill=255)
        glyph = glyph.crop(glyph.getbbox())
        # vertical scale
        sy = random.uniform(*SCALE_Y_RANGE)
        glyph = glyph.resize(
            (glyph.size[0], int(glyph.size[1]*sy)),
            Image.BILINEAR
        )
        # rotate
        glyph = glyph.rotate(
            rand_angle(i), expand=True, resample=Image.BILINEAR
        ).crop(Image.new("L", glyph.size, 0).getbbox())
        # record box
        x0 = int(round(x))
        boxes.append((x0, x0 + glyph.size[0]))
        # paste
        y0 = pad_h + big_h//2 - glyph.size[1]//2
        canvas.paste(glyph, (x0, y0), glyph)
        # advance
        x += glyph.size[0] + random.uniform(*SPACING_RANGE) * RENDER_SCALE

    return np.array(canvas), boxes

def sine_remap(arr, amp, period, phi):
    h, w = arr.shape
    xs = np.arange(w, dtype=np.float32)
    shift = (amp * RENDER_SCALE * np.sin(2*np.pi*xs/(period*RENDER_SCALE) + phi)
            ).astype(np.float32)
    map_x = np.tile(xs, (h,1))
    map_y = (np.arange(h)[:,None] - shift).astype(np.float32)
    return cv2.remap(arr, map_x, map_y,
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def gen_one(text):
    pad = (A + 4) * RENDER_SCALE
    raw, boxes = render_flat(text, pad)

    # глобальная волна
    phi = random.random() * 2 * math.pi
    warped = sine_remap(raw, A, P, phi)

    # локальная волна для каждой буквы
    for x0, x1 in boxes:
        region = warped[:, x0:x1]
        phi_l = random.random() * 2 * math.pi
        warped[:, x0:x1] = sine_remap(region, LOCAL_A, LOCAL_P, phi_l)

    # tight crop
    ys, xs = np.where(warped)
    cropped = warped[ys.min():ys.max()+1, xs.min():xs.max()+1]

    # final downsample once with high-quality lanczos
    pil = Image.fromarray(cropped)
    pil = pil.resize((IMG_W, IMG_H), resample=Image.LANCZOS)
    bw  = np.array(pil)
    _, bw = cv2.threshold(bw, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if PADDING > 0:
        bw = cv2.copyMakeBorder(
            bw,
            PADDING, PADDING,    # top, bottom
            PADDING, PADDING,    # left, right
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
    return bw

def main(n=N_IMAGES):
    # 1) Сканируем существующие файлы и вытаскиваем индексы
    pat = re.compile(r"^captcha_(\d{4})_.*\.(?:png|jpe?g)$")
    existing = []
    for p in OUT_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            existing.append(int(m.group(1)))
    start = (max(existing) + 1) if existing else 0

    # 2) Генерируем от start до start+n-1
    for offset in range(n):
        idx = start + offset
        txt = "".join(random.choice(ALPHABET) for _ in range(CAPTCHA_LEN))
        img = gen_one(txt)
        # если нужен PADDING
        if PADDING:
            img = cv2.copyMakeBorder(
                img,
                PADDING,PADDING,PADDING,PADDING,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
        fname = OUT_DIR / f"captcha_{idx:04d}_{txt}.png"
        cv2.imwrite(str(fname), img)
        print(f"{idx:04d} → {fname.name}")

    print("Saved", n, "new images to", OUT_DIR.resolve())
if __name__ == "__main__":
    main()
