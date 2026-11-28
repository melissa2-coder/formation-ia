"""
Générateur de vidéo pédagogique « Apprendre l'IA » (stage / onboarding).
Rendu : 1280x720 @ 24 fps, ~90 s, encodage mp4v via OpenCV.
Usage : python3 make_video.py [fichier_sortie.mp4]
"""

import math
import os
import random
import sys

import cv2
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
FPS = 24
FADE = 0.7

import matplotlib
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
F_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
F_REG = os.path.join(FONT_DIR, "DejaVuSans.ttf")


def font(size, bold=True):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def rgba_text(text, fnt, fill, pad=6):
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = dummy.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), text, font=fnt, fill=fill)
    return img


def rgba_chip(text, fnt, accent, pad_x=16, pad_y=8):
    txt = rgba_text(text, fnt, (*accent, 255))
    tw, th = txt.size
    img = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [0, 0, img.width - 1, img.height - 1],
        radius=(img.height) // 2,
        fill=(*accent, 36),
        outline=(*accent, 220),
        width=2,
    )
    img.alpha_composite(txt, (pad_x, pad_y))
    return img


def badge(n, accent, size=54):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=14, fill=(*accent, 60), outline=(*accent, 235), width=2)
    f = font(int(size * 0.5), True)
    txt = rgba_text(n, f, (*accent, 255), pad=0)
    img.alpha_composite(txt, ((size - txt.width) // 2, (size - txt.height) // 2))
    return img


def gradient_base(accent, top=(15, 19, 40), bottom=(5, 7, 17)):
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    a = np.array(top, np.float32)[None, None, :]
    b = np.array(bottom, np.float32)[None, None, :]
    arr = a * (1 - t) + b * t
    arr = np.repeat(arr, W, axis=1)
    return arr


def neural_bg(accent, seed=1):
    rng = random.Random(seed)
    arr = gradient_base(accent)
    acc = np.array(accent, np.float32)

    nodes = []
    for _ in range(95):
        nodes.append((rng.uniform(0.02, 0.98), rng.uniform(0.04, 0.96), rng.uniform(2.0, 4.6)))

    eimg = Image.new("RGB", (W, H), (0, 0, 0))
    ed = ImageDraw.Draw(eimg)
    for i, (x1, y1, r1) in enumerate(nodes):
        for x2, y2, r2 in nodes[i + 1:]:
            d = math.hypot((x1 - x2) * W, (y1 - y2) * H)
            if d < 210:
                a = int(90 * (1 - d / 210))
                if a > 12:
                    col = tuple(int(c * a) for c in acc)
                    ed.line([(x1 * W, y1 * H), (x2 * W, y2 * H)], fill=col)
    edges = np.asarray(eimg, np.float32)
    eimg = eimg.filter(ImageFilter.GaussianBlur(1.4))
    edges = np.asarray(eimg, np.float32)

    nimg = Image.new("RGB", (W, H), (0, 0, 0))
    nd = ImageDraw.Draw(nimg)
    for x, y, r in nodes:
        col = tuple(int(c * 0.30) for c in acc)
        nd.ellipse([x * W - r * 3.2, y * H - r * 3.2, x * W + r * 3.2, y * H + r * 3.2], fill=col)
    nimg = nimg.filter(ImageFilter.GaussianBlur(7))
    nd2 = ImageDraw.Draw(nimg)
    for x, y, r in nodes:
        nd2.ellipse([x * W - r, y * H - r, x * W + r, y * H + r], fill=tuple(int(c * 0.85) for c in acc))

    arr = arr + edges * 0.55 + np.asarray(nimg, np.float32) * 0.9
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    vignette = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vignette)
    vd.ellipse([-W * 0.35, -H * 0.45, W * 1.35, H * 1.45], fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(180))
    va = np.asarray(vignette, np.float32)[:, :, None] / 255.0
    dark = np.array([0.25, 0.2, 0.42], np.float32)[None, None, :]
    out = np.asarray(img, np.float32) * va + np.asarray(img, np.float32) * (1 - va) * dark
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def ken_burns(base, p):
    s = 1.0 + 0.06 * ease(p)
    nw, nh = int(W * s) + 2, int(H * s) + 2
    im = base.resize((nw, nh), Image.BILINEAR)
    x = (nw - W) // 2
    y = (nh - H) // 2
    return im.crop((x, y, x + W, y + H))


def paste(img, layer, xy, alpha):
    if alpha <= 0.01:
        return
    layer = layer.copy()
    if alpha < 0.995:
        a = layer.getchannel(3).point(lambda v: int(v * alpha))
        layer.putalpha(a)
    img.alpha_composite(layer, (int(xy[0]), int(xy[1])))


def style_line(draw, x0, y, x1, accent, width=3, alpha=255):
    draw.rounded_rectangle([x0, y, x1, y + width], radius=width // 2, fill=(*accent, alpha))


class ChapterSlide:
    def __init__(self, kicker, title, bullets, accent, seed, dur, index, total, fs=34, gap=24, bwd=54):
        self.dur = dur
        self.accent = accent
        self.base = neural_bg(accent, seed)
        self.kicker = rgba_chip(kicker, font(26, True), accent)
        self.title = rgba_text(title, font(62, True), (240, 244, 255, 255))
        tline = Image.new("RGBA", (self.title.width, 6), (0, 0, 0, 0))
        d = ImageDraw.Draw(tline)
        d.rounded_rectangle([0, 0, self.title.width - 1, 5], radius=3, fill=(*accent, 235))
        self.tline = tline
        self.counter = rgba_text(f"{index:02d} / {total:02d}", font(24, True), (150, 160, 190, 220), pad=2)
        self.bullets = []
        for i, b in enumerate(bullets):
            bimg = badge(f"{i + 1:02d}", accent, bwd)
            lines = wrap(b, font(fs, False), W - 320)
            block_h = sum(l.height + 6 for l in lines)
            comp = Image.new("RGBA", (bimg.width + 22 + max(l.width for l in lines), block_h), (0, 0, 0, 0))
            comp.alpha_composite(bimg, (0, max(0, (block_h - bimg.height) // 2)))
            y = 0
            for l in lines:
                comp.alpha_composite(l, (bimg.width + 22, y))
                y += l.height + 6
            self.bullets.append(comp)
        self.gap = gap
        layout(self)

    def frame(self, lt):
        img = ken_burns(self.base, lt / self.dur).convert("RGBA")
        al = ease((lt - 0.10) / 0.55)
        paste(img, self.kicker, (self.kx, self.ky), al)
        al = ease((lt - 0.35) / 0.65)
        paste(img, self.title, (self.tx, self.ty), al)
        al = ease((lt - 0.60) / 0.50)
        paste(img, self.tline, (self.tx, self.ty + self.title.height + 8), al)
        paste(img, self.counter, (W - 130, 42), ease((lt - 0.2) / 0.5))
        for i, b in enumerate(self.bullets):
            t0 = 1.00 + i * 0.95
            a = ease((lt - t0) / 0.75)
            dy = 34 * (1 - a)
            paste(img, b, (self.bx, self.bys[i] + dy), a)
        return img


def wrap(text, fnt, maxw):
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    words = text.split()
    lines = []
    cur = ""
    for w0 in words:
        test = (cur + " " + w0).strip()
        if dummy.textlength(test, font=fnt) <= maxw - 24:
            cur = test
        else:
            lines.append(rgba_text(cur, fnt, (212, 220, 242, 255), pad=12))
            cur = w0
    if cur:
        lines.append(rgba_text(cur, fnt, (212, 220, 242, 255), pad=12))
    return lines


def layout(slide):
    y = 58
    slide.kx = 84
    slide.ky = y
    y += slide.kicker.height + 26
    slide.tx = 84
    slide.ty = y
    y += slide.title.height + 14 + 30
    slide.bx = 84
    slide.bys = []
    for b in slide.bullets:
        slide.bys.append(y)
        y += b.height + slide.gap
    return slide


class IntroSlide:
    def __init__(self, accent, seed, dur):
        self.dur = dur
        self.accent = accent
        self.base = neural_bg(accent, seed)
        self.title = rgba_text("APPRENDRE  L'IA", font(96, True), (245, 248, 255, 255))
        self.sub = rgba_text("Les essentiels pour bien démarrer ton stage", font(40, False), (185, 197, 230, 255))
        self.chip = rgba_chip("FORMATION · 90 SECONDES", font(26, True), accent)
        self.particles = []
        rng = random.Random(7)
        for _ in range(60):
            self.particles.append([rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.8, 2.6), rng.uniform(10, 34)])

    def frame(self, lt):
        p = lt / self.dur
        img = ken_burns(self.base, p).convert("RGBA")
        d = ImageDraw.Draw(img)
        for x, y, r, sp in self.particles:
            cy = (y - sp * lt) % (H + 40) - 20
            tw = 0.5 + 0.5 * math.sin(lt * 1.7 + x * 0.05)
            a = int(150 * tw * (0.4 + 0.6 * min(1, lt / 2)))
            d.ellipse([x - r, cy - r, x + r, cy + r], fill=(*self.accent, a))
        sc = 0.94 + 0.06 * ease(lt / 1.4)
        tw_ = int(self.title.width * sc)
        th_ = int(self.title.height * sc)
        t_img = self.title.resize((tw_, th_), Image.BILINEAR)
        paste(img, t_img, ((W - tw_) // 2, 224), ease(lt / 1.2))
        lw = int(ease((lt - 0.7) / 1.1) * 460)
        x0 = (W - 460) // 2
        if lw > 3:
            d.rounded_rectangle([x0, 336, x0 + lw, 342], radius=3, fill=(*self.accent, 235))
        paste(img, self.sub, ((W - self.sub.width) // 2, 366), ease((lt - 1.3) / 0.9))
        paste(img, self.chip, ((W - self.chip.width) // 2, 448), ease((lt - 2.0) / 0.9))
        return img


class OutroSlide:
    def __init__(self, accent, seed, dur):
        self.dur = dur
        self.accent = accent
        self.base = neural_bg(accent, seed)
        self.title = rgba_text("À toi de jouer !", font(84, True), (245, 248, 255, 255))
        self.lines = [
            rgba_text("Pratique un peu chaque jour", font(36, True), (52, 211, 153, 255), pad=8),
            rgba_text("Construis un mini-projet dont tu parles", font(36, True), (56, 189, 248, 255), pad=8),
            rgba_text("Pose plein de questions", font(36, True), (167, 139, 250, 255), pad=8),
        ]
        self.end = rgba_text("Bonne chance pour ton stage — La meilleure façon d'apprendre l'IA : la pratiquer.", font(28, False), (170, 182, 215, 255))

    def frame(self, lt):
        img = ken_burns(self.base, lt / self.dur).convert("RGBA")
        sc = 0.95 + 0.05 * ease(lt / 1.3)
        tw_ = int(self.title.width * sc)
        t_img = self.title.resize((tw_, int(self.title.height * sc)), Image.BILINEAR)
        paste(img, t_img, ((W - tw_) // 2, 150), ease(lt / 1.1))
        y = 320
        for i, l in enumerate(self.lines):
            paste(img, l, ((W - l.width) // 2, y), ease((lt - 0.9 - i * 0.6) / 0.8))
            y += l.height + 10
        paste(img, self.end, ((W - self.end.width) // 2, y + 34), ease((lt - 3.0) / 1.0))
        return img


CYAN = (34, 211, 238)
VIOLET = (167, 139, 250)
GREEN = (52, 211, 153)
AMBER = (251, 191, 36)
BLUE = (56, 189, 248)
ORANGE = (251, 146, 60)
PINK = (244, 114, 182)

chapters = [
    ("CHAPITRE 01 · DÉFINITION", "Qu'est-ce que l'IA ?", [
        "Des logiciels qui apprennent à partir de données",
        "Ils repèrent des régularités pour faire des prédictions",
        "Plus ils voient d'exemples, plus ils s'améliorent",
    ], CYAN, 11, 9),
    ("CHAPITRE 02 · MÉCANIQUE", "Comment une machine apprend", [
        "On collecte des données : le « dataset »",
        "Le modèle cherche les régularités : l'entraînement",
        "On mesure les erreurs, on ajuste, on recommence",
        "Une fois entraîné, il prédit sur des données nouvelles",
    ], VIOLET, 22, 11),
    ("CHAPITRE 03 · PAYSAGE", "Les grandes familles d'IA", [
        "ML supervisé : classer un e-mail, prévoir un prix",
        "Deep learning : analyser images, sons et textes",
        "IA générative : rédiger du texte, du code, des images",
    ], GREEN, 33, 10),
    ("CHAPITRE 04 · VOCABULAIRE", "Le vocabulaire à connaître", [
        "Dataset : les exemples dont le modèle apprend",
        "Modèle : le programme entraîné qui prédit",
        "Prompt : la consigne donnée à une IA générative",
        "Hallucination : une réponse fausse mais crédible",
    ], AMBER, 44, 10),
    ("CHAPITRE 05 · OUTILS", "Ta boîte à outils de stagiaire", [
        "Python : le langage n° 1 de l'IA",
        "Google Colab : pour t'entraîner sans rien installer",
        "scikit-learn : le ML classique, simple et efficace",
        "PyTorch : le standard du deep learning",
        "Le Chat, ChatGPT, Claude : des tuteurs à qui tout demander",
    ], PINK, 55, 12, (30, 12, 46)),
    ("CHAPITRE 06 · PARCOURS", "Ton plan sur 4 semaines", [
        "Semaine 1 : Python et manipulation de données (pandas)",
        "Semaine 2 : premiers modèles avec scikit-learn",
        "Semaine 3 : deep learning et projet encadré",
        "Semaine 4 : mini-projet et restitution à l'équipe",
    ], BLUE, 66, 11),
    ("CHAPITRE 07 · BONNES PRATIQUES", "Réussir ton stage avec l'IA", [
        "Vérifie toujours ce que produit une IA",
        "Ne colle jamais de données confidentielles",
        "Note tes essais, garde tes résultats",
        "Pose des questions : c'est le rôle du stagiaire",
    ], ORANGE, 77, 10),
]

slides = [IntroSlide(CYAN, 1, 8)]
n = len(chapters)
for i, (k, t, b, acc, seed, dur, *opts) in enumerate(chapters):
    kwargs = dict(zip(("fs", "gap", "bwd"), opts[0])) if opts else {}
    slides.append(layout(ChapterSlide(k, t, b, acc, seed, dur, i + 1, n, **kwargs)))
slides.append(OutroSlide(VIOLET, 99, 9))

starts = []
t0 = 0.0
for s in slides:
    starts.append(t0)
    t0 += s.dur - FADE
TOTAL = t0 + FADE


def render_frame(t):
    actives = []
    for s, st in zip(slides, starts):
        if st <= t <= st + s.dur:
            actives.append((s, st))
    if len(actives) == 1:
        s, st = actives[0]
        img = s.frame(t - st).convert("RGBA")
    else:
        (s1, st1), (s2, st2) = actives
        img = s1.frame(t - st1).convert("RGBA")
        img2 = s2.frame(0 if t - st2 <= 0 else t - st2).convert("RGBA")
        w2 = ease((t - st2) / FADE)
        img = Image.blend(img, img2, w2)
    d = ImageDraw.Draw(img)
    prog = clamp(t / TOTAL)
    d.rounded_rectangle([0, H - 7, int(W * prog), H - 2], radius=2, fill=(*CYAN, 255))
    d.rounded_rectangle([int(W * prog), H - 7, W - 1, H - 2], radius=2, fill=(28, 34, 58, 200))
    return img


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "apprendre_lia.mp4"
    print(f"TOTAL = {TOTAL:.2f}s, frames = {int(TOTAL * FPS)}", flush=True)
    params = [cv2.VIDEOWRITER_PROP_QUALITY, 88]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H), params)
    if not writer.isOpened():
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    nfr = int(TOTAL * FPS)
    for i in range(nfr):
        t = i / FPS
        frame = render_frame(t)
        bgr = np.asarray(frame.convert("RGB"))[:, :, ::-1].copy()
        writer.write(bgr)
        if i % 120 == 0:
            print(f"frame {i}/{nfr} (t={t:.1f}s)", flush=True)
    writer.release()
    poster = render_frame(3.6).convert("RGB")
    poster.save("poster.jpg", quality=90)

    try:
        import imageio_ffmpeg
        from soundtrack import make_soundtrack
        make_soundtrack(TOTAL)
        tmp = out + ".tmp.mp4"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", out, "-i", "soundtrack.wav",
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "160k", "-shortest", tmp]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, out)
        print("audio muxed", out, flush=True)
    except ImportError:
        print("video muette (pip install imageio-ffmpeg pour ajouter la bande-son)", flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
