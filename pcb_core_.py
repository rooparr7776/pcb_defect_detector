"""Classical-CV PCB inspection core (no ML / no deep learning).

Idea: every board of this design should look the same, so for each component
location we compare the test board against a LIBRARY of good (golden) boards
instead of a single reference. A present component matches at least one golden
board very well; a missing one matches none of them. XML files are used only
(a) to read the component boxes of the canonical reference and
(b) afterwards, for evaluation.
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np

EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
SCALE = 0.5          # working resolution (0.5 = half size, ~4x faster)
ECC_SCALE = 0.4      # extra downscale (relative to SCALE image) used for ECC


# ----------------------------------------------------------------- dataset
def read_xml(img_path):
    """{name: {'box': (x1,y1,x2,y2), 'rotated': 0/1}} for an image's XML."""
    xml = Path(img_path).with_suffix('.xml')
    out = {}
    if not xml.exists():
        return out
    root = ET.parse(str(xml)).getroot()
    for ob in root.findall('object'):
        # some XMLs use <name>, this dataset uses <n>
        name = (ob.findtext('name') or ob.findtext('n') or '').strip()
        b = ob.find('bndbox')
        if not name or b is None:
            continue
        try:
            box = tuple(int(float(b.findtext(t))) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            rot = int(float(ob.findtext('rotated', '0') or 0))
        except (TypeError, ValueError):
            continue
        if box[2] > box[0] and box[3] > box[1]:
            out[name] = {'box': box, 'rotated': rot}
    return out


def discover(folder):
    """Annotated images directly inside the dataset folder (skips 'errors')."""
    p = Path(folder)
    if not p.exists():
        return []
    files = [f for f in p.iterdir()
             if f.suffix.lower() in EXT and f.with_suffix('.xml').exists()]
    return sorted(files, key=lambda f: (f.stem.split('.')[0].zfill(6), f.name))


def is_golden(path):
    """Golden board = plain numeric name (e.g. 12.jpg), 60 components, none rotated."""
    if not re.fullmatch(r'\d+', Path(path).stem):
        return False
    o = read_xml(path)
    return len(o) == 60 and not any(v['rotated'] for v in o.values())


def board_id(path):
    return Path(path).stem.split('.')[0]


# ------------------------------------------------------------ preprocessing
def load_small(path, scale=SCALE):
    a = cv2.imread(str(path))
    if a is None:
        return None
    return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def to_gray_blur(a):
    return cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def align(ref_bgr, test_bgr):
    """Affine ECC alignment of test onto ref. Returns (aligned, corr, ok)."""
    h, w = ref_bgr.shape[:2]
    if test_bgr.shape[:2] != (h, w):
        test_bgr = cv2.resize(test_bgr, (w, h))
    g = lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
                             None, fx=ECC_SCALE, fy=ECC_SCALE).astype('float32') / 255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        corr, warp = cv2.findTransformECC(
            g(ref_bgr), g(test_bgr), warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        warp[:, 2] /= ECC_SCALE
        out = cv2.warpAffine(test_bgr, warp, (w, h),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        return out, float(corr), True
    except cv2.error:
        return test_bgr, 0.0, False


def channels(bgr):
    """Two blurred views used for matching: brightness and Lab colour 'a' (green<->red)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB)
    return {'g': to_gray_blur(bgr), 'a': lab[:, :, 1]}


class Library:
    """Golden boards aligned to one canonical board."""

    def __init__(self, golden_paths, canonical_path, scale=SCALE):
        self.scale = scale
        self.canonical_path = Path(canonical_path)
        self.canon_bgr = load_small(canonical_path, scale)
        self.objects = read_xml(canonical_path)          # canonical ROI boxes
        self.ch = {}                                     # board id -> channels
        for p in golden_paths:
            bid = board_id(p)
            img = self.canon_bgr if p == canonical_path else load_small(p, scale)
            if img is None:
                continue
            if p != canonical_path:
                img, _, _ = align(self.canon_bgr, img)
            self.ch[bid] = channels(img)

    @property
    def gray(self):                                      # (kept for the UI: library size)
        return self.ch


# --------------------------------------------------------------- matching
def best_match(template_gray, test_gray, box_s, search):
    """Normalised cross-correlation of template inside a small search window."""
    x1, y1, x2, y2 = box_s
    h, w = test_gray.shape
    tpl = template_gray[y1:y2, x1:x2]
    if tpl.size == 0 or float(tpl.std()) < 2.0:
        return None
    sx, sy = max(0, x1 - search), max(0, y1 - search)
    ex, ey = min(w, x2 + search), min(h, y2 + search)
    area = test_gray[sy:ey, sx:ex]
    if area.shape[0] < tpl.shape[0] or area.shape[1] < tpl.shape[1]:
        return None
    res = cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(res)[1])


def inspect(lib, test_path, miss_thr=0.58, defect_thr=0.70, search=24,
            exclude_own_board=True, test_bgr=None):
    """Score every component. Score = mean of (gray NCC, colour NCC), each taken as the
    BEST match over all golden boards. 1.0 = looks like a good board; low = defect."""
    s = lib.scale
    test = test_bgr if test_bgr is not None else load_small(test_path, s)
    aligned, corr, ok = align(lib.canon_bgr, test)
    tc = channels(aligned)
    own = board_id(test_path)
    refs = {k: v for k, v in lib.ch.items() if not (exclude_own_board and k == own)}
    rows = []
    for name, meta in lib.objects.items():
        box_s = tuple(int(v * s) for v in meta['box'])
        per = {}
        for c in ('g', 'a'):
            v = [(best_match(g[c], tc[c], box_s, search), k) for k, g in refs.items()]
            v = [x for x in v if x[0] is not None]
            if v:
                per[c] = max(v)
        if len(per) < 2:
            rows.append(dict(Component=name, Score=np.nan, Status='SKIPPED (flat ROI)',
                             Gray_match=None, Colour_match=None, Best_golden=None, box=box_s))
            continue
        sc = 0.5 * (per['g'][0] + per['a'][0])
        if sc < miss_thr:
            status = 'MISSING'
        elif sc < defect_thr:
            status = 'ORIENTATION/PLACEMENT SUSPECT'
        else:
            status = 'OK'
        rows.append(dict(Component=name, Score=round(sc, 3), Status=status,
                         Gray_match=round(per['g'][0], 3), Colour_match=round(per['a'][0], 3),
                         Best_golden=per['g'][1], box=box_s))
    return dict(rows=rows, aligned=aligned, ecc=corr, ecc_ok=ok, test=test)


def draw(aligned, rows):
    out = aligned.copy()
    colors = {'MISSING': (0, 0, 255), 'ORIENTATION/PLACEMENT SUSPECT': (0, 165, 255), 'OK': (0, 200, 0)}
    for r in rows:
        if r['Status'] not in colors:
            continue
        x1, y1, x2, y2 = r['box']
        c = colors[r['Status']]
        t = 1 if r['Status'] == 'OK' else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), c, t)
        if r['Status'] != 'OK':
            cv2.putText(out, r['Component'], (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, c, 1)
    return out


# -------------------------------------------------------------- evaluation
def metrics(actual, pred):
    actual, pred = set(actual), set(pred)
    tp, fp, fn = len(actual & pred), len(pred - actual), len(actual - pred)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return dict(TP=tp, FP=fp, FN=fn, Precision=p, Recall=r, F1=f)


def accuracy(lib, test_rows, actual):
    """Component-level accuracy: correct decisions / all scored components."""
    flagged = {r['Component'] for r in test_rows if r['Status'] in ('MISSING', 'ORIENTATION/PLACEMENT SUSPECT')}
    scored = {r['Component'] for r in test_rows if not str(r['Status']).startswith('SKIPPED')}
    wrong = len(flagged ^ set(actual)) if actual is not None else 0
    return 1 - wrong / max(1, len(scored))


def ground_truth(lib, test_path):
    t = read_xml(test_path)
    missing = set(lib.objects) - set(t)
    rotated = {k for k, v in t.items() if v['rotated'] == 1}
    return missing, rotated
