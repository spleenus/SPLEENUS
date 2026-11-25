import os
import re
import glob
import urllib.request
import tempfile
import io
import contextlib


TMP_DIR = "/var/tmp/jiehyun_tmp"
os.makedirs(f"{TMP_DIR}/triton_cache", exist_ok=True)
os.environ["TMPDIR"] = TMP_DIR
os.environ["TRITON_CACHE_DIR"] = f"{TMP_DIR}/triton_cache"
tempfile.tempdir = TMP_DIR

# hide GCC warning
_CC = f"{TMP_DIR}/gcc_quiet"
if not os.path.exists(_CC):
    with open(_CC, "w") as f:
        f.write('#!/bin/sh\nexec gcc -w "$@"\n')
    os.chmod(_CC, 0o755)
os.environ["CC"] = _CC

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import SPLEENUS as S
from data.base import resize_image, resize_mask, normalize_image, add_channel_axis

# ---------------------------------------------------------------------------
# Paths and model names
# ---------------------------------------------------------------------------
NNUNET_ROOT = "/raid/mpsych/SPLEENUS/nnunet"
SIM = "/raid/mpsych/SPLEENUS/data/simandseg/preprocessed_data"
JPG_ROOT = "/var/tmp/jiehyun_sam3"                  # JPEG copies of frames for SAM3 (originals untouched)
SAM3_EXTRA = f"{TMP_DIR}/sam3_extra"                # files missing from the pip sam3 install

MEDSAM_FT = "/raid/mpsych/SPLEENUS/runs/medsam3_ft_S/best_lora_weights.pt"
MEDSAM_HNEG = "/raid/mpsych/SPLEENUS/runs/medsam3_ft_S_hneg/best_lora_weights.pt"

NNUNET = {
    "E0":   ("Dataset101_SpleenUS_Spleenex",      101, f"{NNUNET_ROOT}/nnunet_S"),
    "E3":   ("Dataset102_SpleenUS_S_RUSneg_54",   102, f"{NNUNET_ROOT}/nnunet_S_RUSneg_54"),
    "E3x3": ("Dataset103_SpleenUS_S_RUSneg_54x3", 103, f"{NNUNET_ROOT}/nnunet_S_RUSneg_54x3"),
}
LABEL = {"E0": "Spleenex", "E3": "add empty slices", "E3x3": "add empty slices x3"}


import contextlib

def _no_autocast():
    """Only acts if autocast is on (e.g. left on by SAM3)."""
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_image(path):
    return add_channel_axis(normalize_image(resize_image(np.array(Image.open(path).convert("L")))))


def load_pairs(root):
    """root/images and root/labels (png). Returns names, X (N,H,W,1), y (N,H,W,1)."""
    names = sorted(f for f in os.listdir(f"{root}/images") if f.lower().endswith(".png"))
    labels = sorted(f for f in os.listdir(f"{root}/labels") if f.lower().endswith(".png"))
    assert names == labels, "image and label names do not match"
    X, y = [], []
    for n in names:
        X.append(_load_image(f"{root}/images/{n}"))
        msk = np.array(Image.open(f"{root}/labels/{n}").convert("L"))
        msk = ((msk > 0) * 255).astype(np.uint8)                # 0/1 labels -> 0/255
        y.append(add_channel_axis((resize_mask(msk) > 127).astype(np.float32)))
    return names, np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_set(name):
    key = {"D": "DeepSPV", "R": "Roboflow"}[name]
    S.DATA.setup(train=["Spleenex"], test=[key], val_fraction=0.2, randomize=True, seed=42)
    return S.get_test()


def load_video(frames_dir):
    files = {}
    for f in glob.glob(f"{frames_dir}/*"):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            nums = re.findall(r"\d+", os.path.splitext(os.path.basename(f))[0])
            if nums:
                files[int(nums[-1])] = f
    ids = sorted(files)
    return ids, np.array([_load_image(files[i]) for i in ids], dtype=np.float32)  # Returns ids, X (N,H,W,1)


def _name(frames_dir):
    return os.path.basename(frames_dir.rstrip("/"))


# ---------------------------------------------------------------------------
# Models (each one is loaded once and cached)
# ---------------------------------------------------------------------------
_CACHE = {}


def load_model(key="E0"):
    """nnUNet model: 'E0', 'E3' or 'E3x3'."""
    if key not in _CACHE:
        dn, did, base = NNUNET[key]
        m = S.MODELS.get("nnunet", dataset_name=dn, dataset_id=did)
        m.setup(base)
        m.load(folds=(0, 1, 2, 3, 4))
        _CACHE[key] = m
    return _CACHE[key]


def nn_predict(key, X):
    return np.asarray(load_model(key).predict(X)).reshape(X.shape[:3]) > 0.5


def get_medsam(weights=None):
    key = weights or "medsam_base"
    if key not in _CACHE:
        m = S.MODELS.get("medsam3")
        m.load(weights_path=weights)
        _CACHE[key] = m
    return _CACHE[key]


def medsam_predict(model, X):
    with _no_autocast():
        return np.asarray(model.predict(X)).reshape(X.shape[:3]) > 0.5


def evaluate(model, X, y):
    with _no_autocast():
        return model.evaluate(X, y)


def visualize(model, X, y, n=5):
    with _no_autocast():
        model.visualize(X, y, n=n)


def free_models():
    """Drop all cached models and free GPU memory."""
    _CACHE.clear()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _grid(name, frame_ids, imgs, cols, row_h=3.2):
    if matplotlib.get_backend().lower() == "agg": 
        try:
            plt.switch_backend("module://matplotlib_inline.backend_inline")
        except Exception:
            pass
    n, c = len(frame_ids), 1 + len(cols)
    fig, axes = plt.subplots(n, c, figsize=(row_h * c, row_h * n))
    axes = np.atleast_2d(axes)
    for r, fid in enumerate(frame_ids):
        axes[r, 0].imshow(imgs[r], cmap="gray")
        axes[r, 0].set_title(f"{name} {fid}  Original", fontsize=9)
        for k, (title, M) in enumerate(cols, start=1):
            m = M[r].astype(float)
            axes[r, k].imshow(imgs[r], cmap="gray")
            axes[r, k].imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5, vmin=0, vmax=1)
            axes[r, k].set_title(title, fontsize=9)
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout()
    plt.show()


def show(V, ids=None, row_h=3.2):
    all_ids = V["ids"]
    if ids is None:
        ids = [all_ids[j] for j in np.linspace(0, len(all_ids) - 1, 7).astype(int)]
    idx = [all_ids.index(i) for i in ids]
    cols = [(title, M[idx]) for title, M in V["cols"].items()]
    _grid(V["name"], ids, V["X"][idx, ..., 0], cols, row_h)


def show_all(V, per_fig=10, row_h=2.4):
    for s in range(0, len(V["ids"]), per_fig):
        show(V, V["ids"][s:s + per_fig], row_h)


# ---------------------------------------------------------------------------
# 1. sim&seg spleen slices (nnUNet)
# ---------------------------------------------------------------------------
def eval_pairs(root, model="E0"):
    names, X, y = load_pairs(root)
    pred = nn_predict(model, X)
    dices = [_dice(p, t[..., 0] > 0.5) for p, t in zip(pred, y)]
    print(f"[{LABEL[model]}]")
    for n, d in zip(names, dices):
        print(f"  {n:10s} {d:.3f}")
    print(f"  mean dice ({len(names)} slices): {np.mean(dices):.3f}")
    return dices


def show_pairs(root, model="E0"):
    names, X, y = load_pairs(root)
    load_model(model).visualize(X, y, n=len(names))


# ---------------------------------------------------------------------------
# 2-3. Public sets D and R (nnUNet)
# ---------------------------------------------------------------------------
def eval_sets(models=("E0", "E3", "E3x3")):
    sets = {n: load_set(n) for n in ("D", "R")}
    print(f"{'model':22s} {'D':>7s} {'R':>7s} {'D+R':>7s} | empty slices# D / R")
    for k in models:
        st = {}
        for n, (X, y) in sets.items():
            p = nn_predict(k, X).reshape(len(X), -1)
            g = y.reshape(len(y), -1) > 0.5
            st[n] = (2 * (p & g).sum(), p.sum() + g.sum(), 100 * np.mean(p.sum(1) == 0))
        dr = (st["D"][0] + st["R"][0]) / max(st["D"][1] + st["R"][1], 1)
        print(f"{LABEL[k]:22s} {st['D'][0] / max(st['D'][1], 1):7.3f} {st['R'][0] / max(st['R'][1], 1):7.3f} "
              f"{dr:7.3f} | {st['D'][2]:.0f} / {st['R'][2]:.0f}")


def show_set(name="D", n=3, seed=42, models=("E0", "E3", "E3x3")):
    X, y = load_set(name)
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    preds = {k: nn_predict(k, X[idx]) for k in models}
    fig, axes = plt.subplots(n, 2 + len(models), figsize=(3.2 * (2 + len(models)), 3.2 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        t = y[i, ..., 0] > 0.5
        axes[r, 0].imshow(X[i, ..., 0], cmap="gray"); axes[r, 0].set_title(f"{name} #{i}")
        axes[r, 1].imshow(t, cmap="gray"); axes[r, 1].set_title("GT")
        for c, k in enumerate(models, start=2):
            p = preds[k][r]
            axes[r, c].imshow(p, cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"{LABEL[k]}  Dice {_dice(p, t):.2f}", fontsize=9)
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Pilot video, nnUNet only
# ---------------------------------------------------------------------------
def run_nnunet_video(frames_dir, models=("E0", "E3", "E3x3")):
    ids, X = load_video(frames_dir)
    cols = {LABEL[k]: nn_predict(k, X) for k in models}
    return dict(name=_name(frames_dir), ids=ids, X=X, cols=cols)


# ---------------------------------------------------------------------------
# 5. SAM3 tracker: start mask on the middle frame, propagate forward and backward
# ---------------------------------------------------------------------------
def _ensure_bpe():
    import sam3
    p = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz", p)
    return p


def _reset_autocast():
    try:
        torch.set_autocast_enabled("cuda", False); torch.set_autocast_dtype("cuda", torch.float16)
    except TypeError:                                   # older torch API
        torch.set_autocast_enabled(False); torch.set_autocast_gpu_dtype(torch.float16)

def _sam3_video_model():
    from sam3.model_builder import build_sam3_video_model
    m = build_sam3_video_model(load_from_HF=True, bpe_path=_ensure_bpe())
    _reset_autocast()
    return m


def get_tracker():
    """Plain SAM3 tracker. It borrows the detector's image backbone so it can read frames."""
    if "sam3_tracker" not in _CACHE:
        m = _sam3_video_model()
        m.tracker.backbone = m.detector.backbone
        _CACHE["sam3_tracker"] = m
    return _CACHE["sam3_tracker"].tracker


def write_jpegs(name, X):
    """SAM3 reads a folder of JPEG frames (00000.jpg, ...). Written once per video."""
    d = f"{JPG_ROOT}/{name}"
    os.makedirs(d, exist_ok=True)
    if len(glob.glob(f"{d}/*.jpg")) != len(X):
        for f in glob.glob(f"{d}/*.jpg"):
            os.remove(f)
        for i, x in enumerate(X[..., 0]):
            Image.fromarray((x * 255).astype(np.uint8)).convert("RGB").save(f"{d}/{i:05d}.jpg", quality=95)
    return d


def track_from_mask(jpg_dir, start_mask, anchor):
    """SAM3 tracker from start_mask on frame `anchor`, forward then backward. (N,H,W) bool, as is."""
    assert start_mask.any(), "start mask is empty"
    tracker = get_tracker()
    N = len(glob.glob(f"{jpg_dir}/*.jpg"))
    H, W = start_mask.shape
    masks = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = tracker.init_state(video_path=jpg_dir)
        tracker.add_new_mask(state, frame_idx=anchor, obj_id=1, mask=torch.as_tensor(start_mask))
        for reverse in (False, True):
            for out in tracker.propagate_in_video(state, anchor, N, reverse, propagate_preflight=True):
                m = next(o for o in out[1:] if torch.is_tensor(o) and o.dim() >= 2 and tuple(o.shape[-2:]) == (H, W))
                masks[int(out[0])] = (m.reshape(-1, H, W)[0] > 0).cpu().numpy()
    return np.stack([masks.get(i, np.zeros((H, W), bool)) for i in range(N)])


def run_tracking(frames_dir, anchor=None):
    """nnUNet Spleenex | SAM3 tracker started from each nnUNet model's middle-frame mask."""
    name = _name(frames_dir)
    ids, X = load_video(frames_dir)
    anchor = len(ids) // 2 if anchor is None else anchor
    jpg = write_jpegs(name, X)
    nn = nn_predict("E0", X)
    cols = {"nnUNet Spleenex": nn}
    for k in ("E0", "E3", "E3x3"):
        start = nn[anchor] if k == "E0" else nn_predict(k, X[anchor:anchor + 1])[0]
        cols[f"SAM3 from {LABEL[k]}"] = track_from_mask(jpg, start, anchor)
    print(f"{name}: {len(ids)} frames, start frame {ids[anchor]}")
    return dict(name=name, ids=ids, X=X, cols=cols)


# ---------------------------------------------------------------------------
# 6. MedSAM3 checks
# ---------------------------------------------------------------------------
def count_empty(model, X, names):
    """Spleen-free images: empty prediction = correct. Also split by person (first letter of the name)."""
    empty = ~medsam_predict(model, X).any(axis=(1, 2))
    print(f"empty predictions: {empty.sum()}/{len(X)} | false detections: {(~empty).sum()}/{len(X)}")
    for p in sorted({n[0] for n in names}):
        sel = np.array([n[0] == p for n in names])
        print(f"  person {p}: empty {empty[sel].sum()}/{sel.sum()}")
    return empty


def eval_DR(models):
    """models: {name: MedSAM3 model}. Dice on D and R as one small table (library logs hidden)."""
    rows = {}
    for set_name in ("D", "R"):
        with contextlib.redirect_stdout(io.StringIO()):          # hide loading / evaluate prints here only
            X, y = load_set(set_name)
            for k, m in models.items():
                rows.setdefault(k, {})[set_name] = evaluate(m, X, y)["dice"]
    print(f"{'model':22s} {'D':>7s} {'R':>7s}")
    for k, r in rows.items():
        print(f"{k:22s} {r['D']:7.3f} {r['R']:7.3f}")


def run_text_models(frames_dir, models):
    """models: {name: MedSAM3 model}. Text 'spleen', frame by frame."""
    ids, X = load_video(frames_dir)
    cols = {k: medsam_predict(m, X) for k, m in models.items()}
    return dict(name=_name(frames_dir), ids=ids, X=X, cols=cols)


# ---------------------------------------------------------------------------
# 7. SAM3's own detector + tracker (MedSAM3 FT LoRA in the detector)
# ---------------------------------------------------------------------------
def _add_missing_triton():
    """The pip sam3 install misses sam3/perflib/triton, which detector+tracker needs. Download 2 files once."""
    import sam3.perflib
    d = f"{SAM3_EXTRA}/triton"
    os.makedirs(d, exist_ok=True)
    for f in ("connected_components.py", "nms.py"):
        if not os.path.exists(f"{d}/{f}"):
            urllib.request.urlretrieve(
                f"https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/perflib/triton/{f}", f"{d}/{f}")
    if SAM3_EXTRA not in sam3.perflib.__path__:
        sam3.perflib.__path__.append(SAM3_EXTRA)


def get_det_trk(lora_path=MEDSAM_FT):
    """SAM3 video model (detector + tracker) with the MedSAM3 LoRA loaded into the detector."""
    key = f"det_trk:{lora_path}"
    if key not in _CACHE:
        from models.sam.lora import LoRAConfig, apply_lora_to_model
        _add_missing_triton()
        m = _sam3_video_model()
        apply_lora_to_model(m.detector, LoRAConfig(rank=16, alpha=32, dropout=0.0))
        _, unexpected = m.detector.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)
        assert not unexpected, f"{len(unexpected)} LoRA weights did not match"
        _CACHE[key] = m.to("cuda").eval()
    return _CACHE[key]


def det_trk(model, jpg_dir, anchor, text="spleen"):
    """SAM3 detector+tracker: find `text` in every frame and track it. All found objects are kept, as is."""
    N = len(glob.glob(f"{jpg_dir}/*.jpg"))
    res = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = model.init_state(resource_path=jpg_dir)
        model.add_prompt(state, frame_idx=anchor, text_str=text)
        for reverse in (False, True):
            for f, d in model.propagate_in_video(state, start_frame_idx=anchor,
                                                 max_frame_num_to_track=N, reverse=reverse):
                res[int(f)] = np.asarray(d["out_binary_masks"]).astype(bool)
    H, W = next(iter(res.values())).shape[-2:]
    return np.stack([res[i].any(axis=0) if i in res else np.zeros((H, W), bool) for i in range(N)])


def run_det_trk(frames_dir, lora_path=MEDSAM_FT, anchor=None):
    """MedSAM3 FT | SAM3 tracker only (from MedSAM3's middle mask) | SAM3 detector+tracker."""
    name = _name(frames_dir)
    ids, X = load_video(frames_dir)
    anchor = len(ids) // 2 if anchor is None else anchor
    jpg = write_jpegs(name, X)
    med = medsam_predict(get_medsam(lora_path), X)
    cols = {
        "MedSAM3 FT": med,
        "SAM3 tracker only": track_from_mask(jpg, med[anchor], anchor),
        "SAM3 detector+tracker": det_trk(get_det_trk(lora_path), jpg, anchor),
    }
    print(f"{name}: {len(ids)} frames, start frame {ids[anchor]}")
    return dict(name=name, ids=ids, X=X, cols=cols)