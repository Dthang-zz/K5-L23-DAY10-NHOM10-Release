#!/usr/bin/env python3
"""Validator của Nhóm 10 — Day 10 Lab.

  python validator/validate.py <delivery_dir> --out <errors.json> [--candidates <c.json>] [--rules-out <rules.json>]

Ràng buộc (RULES.md):
- Chỉ thư viện chuẩn (+ scripts/pngutil.py). Không mạng.
- Chỉ đọc thư mục delivery được truyền vào; không đọc _private/, data_source/, annotation COCO.
- Không nhúng tên ảnh / id / đáp án nào vào code.

Đầu ra: {"errors": [{"type": ..., "target": ...}]}
- target = tên ảnh không đuôi; riêng scene_leak có target = scene_id.
- Mỗi ảnh tối đa MỘT lỗi: nếu một ảnh dính nhiều luật, giữ lỗi có ưu tiên cao nhất (xem PRIORITY).

Trạng thái:
- Bước 1.1: missing_image, wrong_hash, guideline_mismatch, scene_leak.
- Bước 1.2: invalid_class, bbox_out_of_bounds, duplicate_box.
- Bước 1.3: invalid_category_id, thing_overlap.
- Pha 2 (TODO): --candidates (đối chiếu detection <-> panoptic).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Cho phép import scripts/pngutil.py (thư viện chuẩn, được RULES cho phép).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pngutil  # noqa: E402

NUM_CLASSES = 80

# --- Ngưỡng luật detection (đơn vị pixel, trừ khi ghi khác) ---
# bbox_out_of_bounds: nhãn YOLO ghi 6 chữ số thập phân => sai số làm tròn ~1e-6 (chuẩn hoá).
# Box lỗi có thể chỉ tràn mép "một chút", nên chỉ tha thứ đúng phần làm tròn số.
OOB_EPS_NORM = 1e-4
# duplicate_box: box lặp = cùng lớp, CÙNG kích thước, tâm lệch 1-2 px.
# Cặp chồng nhau tự nhiên của COCO (IoU tới ~0,66) gần như luôn khác kích thước, nên kiểm
# theo kích thước + độ lệch tâm thay vì IoU.
DUP_SIZE_TOL_PX = 0.5      # |w1-w2| và |h1-h2|
DUP_CENTER_TOL_PX = 3.0    # khoảng cách tâm

# --- Ngưỡng luật panoptic ---
# thing_overlap: area khai > số pixel thật. COCO panoptic ghi area đúng bằng số pixel,
# nên chỉ cần lệch dương là đáng ngờ; để dung sai nhỏ phòng mask bị ghi lại.
OVERLAP_TOL_PX = 0

# --- Ngưỡng đối chiếu detection <-> panoptic (Pha 2) ---
# Ký hiệu: contain(s, b) = số pixel của segment s nằm trong box b / tổng pixel của s.
#          fill(s, b)    = số pixel của s nằm trong b / diện tích b.
# Hiệu chỉnh bằng một script đo ngưỡng riêng (chỉ đọc delivery) trên lô sạch (50 ảnh calib) và
# seed 10007/11/22: ở lô sạch mọi cặp cùng lớp có contain ≥ 0,995, không có swap/merge nào;
# ứng viên drift trên seed đều ≤ 0,83. Ngưỡng đặt lệch về phía recall (đề ẩn có thể lệch nhẹ
# hơn), vì người xem ảnh sẽ loại ca tự nhiên.
MATCH_MIN_IOU = 0.2        # IoU bbox tối thiểu để xét ghép một box với một segment
SAME_CLASS_BONUS = 0.25    # ưu tiên ghép cùng lớp khi IoU gần nhau
SWAP_MIN_IOU = 0.4         # cặp khác lớp chỉ là ứng viên swap khi trùng vị trí rõ
DRIFT_MAX_CONTAIN = 0.95   # cặp cùng lớp: contain thấp hơn => một phần vật rơi ra ngoài box
MISSING_MIN_FRAC = 0.002   # segment thing tối thiểu 0,2% diện tích ảnh mới xét missing_box
MISSING_MAX_COVER = 0.50   # không box cùng lớp nào chứa quá 50% pixel của segment
MERGE_MIN_FILL = 0.35      # box bị segment phủ ít nhất 35% diện tích ...
MERGE_MIN_EXCL = 0.08      # ... và mỗi box chứa RIÊNG (ngoài phần giao hai box) ≥ 8% pixel của segment.
# Đo trên lô sạch: lệch tự nhiên thường là box lồng nhau / vật đứng sát, khi đó segment nằm
# trọn trong một box (phần riêng của box kia ~0). Segment bị gộp thật gồm hai phần tách biệt.

# Thứ tự ưu tiên khi một ảnh dính nhiều lỗi (chỉ số nhỏ = ưu tiên cao).
# Lý do: lỗi mức file che mọi kiểm tra phía sau (thiếu ảnh thì không kiểm được gì;
# hash sai thì không tin được file); lỗi metadata tiếp theo; rồi tới nội dung nhãn
# detection; cuối cùng là panoptic.
PRIORITY = [
    "missing_image",
    "wrong_hash",
    "guideline_mismatch",
    "invalid_class",
    "bbox_out_of_bounds",
    "duplicate_box",
    "invalid_category_id",
    "thing_overlap",
]


# --------------------------------------------------------------------------- nạp dữ liệu
@dataclass
class Delivery:
    root: Path
    manifest: dict[str, dict]          # tên ảnh -> {image, label, mask, image_sha256, ...}
    metadata: dict[str, dict]          # tên ảnh -> {scene_id, split, job_id, guideline_version}
    jobs: dict[str, dict]              # job_id -> {annotator, reviewer, guideline}
    classes: list[str]                 # chỉ số 0-79 -> tên lớp
    panoptic: dict = field(default_factory=dict)
    size: dict[str, tuple[int, int]] = field(default_factory=dict)   # tên ảnh -> (width, height)
    pan_ann: dict[str, dict] = field(default_factory=dict)            # tên ảnh -> annotation panoptic

    @property
    def category_ids(self) -> set[int]:
        return {c["id"] for c in self.panoptic.get("categories", [])}


def load_delivery(root: Path) -> Delivery:
    """Đọc các file mô tả của lô. Chỉ đọc bên trong `root`."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["items"]
    metadata = {m["image"]: m for m in json.loads((root / "metadata.json").read_text(encoding="utf-8"))["items"]}
    with (root / "jobs.csv").open(encoding="utf-8", newline="") as f:
        jobs = {row["job_id"]: row for row in csv.DictReader(f)}
    classes = []
    for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            idx, name = line.split(" ", 1)
            assert int(idx) == len(classes), "classes.txt phải đánh số liên tục từ 0"
            classes.append(name.strip())
    panoptic = json.loads((root / "panoptic.json").read_text(encoding="utf-8"))
    dv = Delivery(root, manifest, metadata, jobs, classes, panoptic)
    # Kích thước ảnh lấy từ panoptic.json (không cần mở JPEG, không cần Pillow).
    by_id = {}
    for im in panoptic.get("images", []):
        name = Path(im["file_name"]).stem
        dv.size[name] = (int(im["width"]), int(im["height"]))
        by_id[im["id"]] = name
    for ann in panoptic.get("annotations", []):
        name = by_id.get(ann.get("image_id")) or Path(ann["file_name"]).stem
        dv.pan_ann[name] = ann
    return dv


def read_labels(path: Path) -> list[tuple[str, list[float]]]:
    """Đọc file YOLO -> [(class_token, [xc, yc, w, h])]. Giữ class dạng chuỗi để luật tự kiểm."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        rows.append((parts[0], [float(v) for v in parts[1:5]]))
    return rows


def parse_class(token: str) -> int | None:
    """Trả về class_id nếu là số nguyên hợp lệ, ngược lại None."""
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if value.is_integer() else None


def sha256_file(path: Path) -> str:
    """sha256 trên BYTES của file (đọc nhị phân), không phải trên text."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- luật mức file
def check_files(dv: Delivery) -> dict[str, set[str]]:
    """missing_image: manifest có ảnh nhưng file không tồn tại.
    wrong_hash: sha256 thật của file ảnh khác image_sha256 ghi trong manifest.

    Trả về {tên ảnh: {loại lỗi}}. Ảnh thiếu thì không kiểm hash.
    """
    found: dict[str, set[str]] = {}
    for name, item in dv.manifest.items():
        img = dv.root / item["image"]
        if not img.is_file():
            found.setdefault(name, set()).add("missing_image")
            continue
        if sha256_file(img) != item["image_sha256"].strip().lower():
            found.setdefault(name, set()).add("wrong_hash")
    return found


# --------------------------------------------------------------------------- luật metadata
def check_guideline(dv: Delivery) -> dict[str, set[str]]:
    """guideline_mismatch: guideline_version ghi ở ảnh khác guideline ghi ở JOB của ảnh (jobs.csv).

    So với job chứ không so với các ảnh khác: job là nguồn sự thật về phiên bản guideline
    người gán đã dùng; metadata từng ảnh mới là chỗ có thể bị ghi sai.
    """
    found: dict[str, set[str]] = {}
    for name, meta in dv.metadata.items():
        job = dv.jobs.get(meta.get("job_id"))
        if job is None:
            print(f"[cảnh báo] {name}: job_id {meta.get('job_id')!r} không có trong jobs.csv", file=sys.stderr)
            continue
        if meta.get("guideline_version", "").strip() != job["guideline"].strip():
            found.setdefault(name, set()).add("guideline_mismatch")
    return found


def check_scene_leak(dv: Delivery) -> list[str]:
    """scene_leak: một cảnh có ảnh nằm ở cả train và val. Trả về danh sách scene_id (target của lỗi)."""
    splits: dict[str, set[str]] = {}
    for meta in dv.metadata.values():
        splits.setdefault(meta["scene_id"], set()).add(meta["split"])
    return sorted(scene for scene, s in splits.items() if len(s) > 1)


# --------------------------------------------------------------------------- luật detection
def check_detection(dv: Delivery) -> dict[str, set[str]]:
    """invalid_class, bbox_out_of_bounds, duplicate_box trên nhãn YOLO (chuẩn hoá 0-1)."""
    found: dict[str, set[str]] = {}
    for name, item in dv.manifest.items():
        label = dv.root / item["label"]
        if not label.is_file():
            continue
        rows = read_labels(label)
        W, H = dv.size.get(name, (None, None))
        kinds = set()
        boxes_px = []  # (class_id, xc, yc, w, h) theo pixel, cho duplicate_box
        for token, (xc, yc, w, h) in rows:
            cls = parse_class(token)
            if cls is None or not 0 <= cls < NUM_CLASSES:
                kinds.add("invalid_class")
                continue
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            if min(x1, y1) < -OOB_EPS_NORM or max(x2, y2) > 1 + OOB_EPS_NORM or w <= 0 or h <= 0:
                kinds.add("bbox_out_of_bounds")
            if W and H:
                boxes_px.append((cls, xc * W, yc * H, w * W, h * H))
        if find_duplicate(boxes_px):
            kinds.add("duplicate_box")
        if kinds:
            found[name] = kinds
    return found


def find_duplicate(boxes: list[tuple[int, float, float, float, float]]) -> bool:
    """Có cặp box cùng lớp, cùng kích thước (± DUP_SIZE_TOL_PX), tâm lệch ≤ DUP_CENTER_TOL_PX?"""
    for i in range(len(boxes)):
        ci, xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, len(boxes)):
            cj, xj, yj, wj, hj = boxes[j]
            if ci != cj:
                continue
            if abs(wi - wj) <= DUP_SIZE_TOL_PX and abs(hi - hj) <= DUP_SIZE_TOL_PX \
                    and math.hypot(xi - xj, yi - yj) <= DUP_CENTER_TOL_PX:
                return True
    return False


# --------------------------------------------------------------------------- luật panoptic
_MASK_CACHE: dict[Path, Counter] = {}


def mask_pixel_counts(path: Path) -> Counter:
    """{segment_id: số pixel} của một mask panoptic; id = R + G*256 + B*65536.

    Giải mã một lần rồi cache (Pha 2 dùng lại). Đếm bằng Counter trên bộ ba (R, G, B)
    lấy bằng slice -> nhanh hơn nhiều so với vòng lặp từng pixel trong Python.
    """
    if path not in _MASK_CACHE:
        _, _, px = pngutil.read_rgb(path)
        rgb = Counter(zip(px[0::3], px[1::3], px[2::3]))
        _MASK_CACHE[path] = Counter({r + g * 256 + b * 65536: n for (r, g, b), n in rgb.items()})
    return _MASK_CACHE[path]


def check_panoptic(dv: Delivery) -> dict[str, set[str]]:
    """invalid_category_id: category_id không có trong categories.
    thing_overlap: area khai trong segments_info lớn hơn số pixel thật của segment trong mask.
    """
    found: dict[str, set[str]] = {}
    valid = dv.category_ids
    for name, item in dv.manifest.items():
        ann = dv.pan_ann.get(name)
        if ann is None:
            continue
        kinds = set()
        segs = ann.get("segments_info", [])
        if any(s.get("category_id") not in valid for s in segs):
            kinds.add("invalid_category_id")
        mask = dv.root / item["mask"]
        if mask.is_file():
            counts = mask_pixel_counts(mask)
            for s in segs:
                if s["id"] == 0:          # id 0 là nền
                    continue
                if s.get("area", 0) - counts.get(s["id"], 0) > OVERLAP_TOL_PX:
                    kinds.add("thing_overlap")
                    break
        if kinds:
            found[name] = kinds
    return found


# --------------------------------------------------------------------------- đối chiếu (Pha 2)
def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU của hai box dạng (x1, y1, x2, y2)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_ids(path: Path) -> tuple[int, int, list[int]]:
    """Mask -> (W, H, danh sách id segment theo từng pixel, hàng trước cột sau)."""
    w, h, px = pngutil.read_rgb(path)
    return w, h, [r + (g << 8) + (b << 16) for r, g, b in zip(px[0::3], px[1::3], px[2::3])]


def cross_check_image(dv: Delivery, name: str) -> list[dict]:
    """Sinh ứng viên lỗi gán nhãn cho một ảnh bằng cách đối chiếu box YOLO với mask panoptic."""
    item, ann = dv.manifest[name], dv.pan_ann.get(name)
    label, mask = dv.root / item["label"], dv.root / item["mask"]
    if ann is None or not label.is_file() or not mask.is_file():
        return []
    W, H, ids = decode_ids(mask)
    seg_px = Counter(ids)
    cats = {c["id"]: c for c in dv.panoptic["categories"]}
    segs = [s for s in ann["segments_info"]
            if cats.get(s["category_id"], {}).get("isthing") and not s.get("iscrowd") and seg_px.get(s["id"], 0) > 0]
    seg_name = {s["id"]: cats[s["category_id"]]["name"] for s in segs}
    seg_box = {s["id"]: (s["bbox"][0], s["bbox"][1], s["bbox"][0] + s["bbox"][2], s["bbox"][1] + s["bbox"][3]) for s in segs}

    # Box detection theo pixel + số pixel của mỗi segment nằm trong box.
    boxes = []
    for token, (xc, yc, w, h) in read_labels(label):
        cls = parse_class(token)
        if cls is None or not 0 <= cls < NUM_CLASSES:
            continue
        x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
        c1, r1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        c2, r2 = min(W, math.ceil(x2)), min(H, math.ceil(y2))
        inside = Counter()
        for row in range(r1, r2):
            inside.update(ids[row * W + c1: row * W + c2])
        boxes.append({"cls": dv.classes[cls], "xyxy": (x1, y1, x2, y2),
                      "area": max(1, (c2 - c1) * (r2 - r1)), "inside": inside})

    def contain(sid: int, b: dict) -> float:
        return b["inside"].get(sid, 0) / seg_px[sid]

    def fill(sid: int, b: dict) -> float:
        return b["inside"].get(sid, 0) / b["area"]

    def rnd(t) -> list[float]:
        return [round(v, 1) for v in t]

    # Ghép 1-1 tham lam: điểm = IoU bbox (+ thưởng nếu cùng lớp).
    pairs = []
    for bi, b in enumerate(boxes):
        for s in segs:
            iou = bbox_iou(b["xyxy"], seg_box[s["id"]])
            if iou >= MATCH_MIN_IOU:
                pairs.append((iou + (SAME_CLASS_BONUS if b["cls"] == seg_name[s["id"]] else 0), iou, bi, s["id"]))
    pairs.sort(reverse=True)
    used_b, used_s, matched = set(), set(), []
    for _, iou, bi, sid in pairs:
        if bi not in used_b and sid not in used_s:
            used_b.add(bi), used_s.add(sid)
            matched.append((iou, bi, sid))

    out = []

    def add(kind, score, reason, b=None, sid=None):
        out.append({"image": name, "type": kind, "score": round(score, 3),
                    "class_det": b["cls"] if b else None, "class_pan": seg_name.get(sid),
                    "box": rnd(b["xyxy"]) if b else None, "seg_bbox": rnd(seg_box[sid]) if sid else None,
                    "seg_id": sid,
                    "reason": reason})

    def in_rect(sid: int, x1: float, y1: float, x2: float, y2: float) -> int:
        c1, r1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        c2, r2 = min(W, math.ceil(x2)), min(H, math.ceil(y2))
        return sum(ids[row * W + c1: row * W + c2].count(sid) for row in range(r1, r2)) if c2 > c1 else 0

    def dominant(sid: int, b: dict) -> bool:
        """sid là segment cùng lớp chiếm nhiều pixel nhất trong box b?"""
        mine = b["inside"].get(sid, 0)
        return all(b["inside"].get(o["id"], 0) <= mine for o in segs
                   if o["id"] != sid and seg_name[o["id"]] == b["cls"])

    # instance_merge: một segment chiếm ưu thế trong HAI box cùng lớp, mỗi box giữ một phần riêng.
    merged = set()
    for s in segs:
        sid = s["id"]
        hits = [b for b in boxes if b["cls"] == seg_name[sid] and fill(sid, b) >= MERGE_MIN_FILL and dominant(sid, b)]
        best = None
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                a, b = hits[i]["xyxy"], hits[j]["xyxy"]
                both = in_rect(sid, max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])) \
                    if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]) else 0
                ex_a = (hits[i]["inside"].get(sid, 0) - both) / seg_px[sid]
                ex_b = (hits[j]["inside"].get(sid, 0) - both) / seg_px[sid]
                score = min(ex_a, ex_b)
                if score >= MERGE_MIN_EXCL and (best is None or score > best[0]):
                    best = (score, hits[i], ex_a, ex_b)
        if best:
            merged.add(sid)
            add("instance_merge", best[0], f"segment {seg_name[sid]} trải trên 2 box {seg_name[sid]} tách biệt; "
                f"phần riêng trong mỗi box {best[2]:.0%} / {best[3]:.0%} pixel", best[1], sid)

    for iou, bi, sid in matched:
        b = boxes[bi]
        if b["cls"] != seg_name[sid]:
            if iou >= SWAP_MIN_IOU:
                add("class_or_category_swap", iou, f"cùng vị trí (IoU {iou:.2f}) nhưng detection ghi "
                    f"'{b['cls']}', panoptic ghi '{seg_name[sid]}': mở ảnh xem bên nào sai", b, sid)
        elif sid not in merged:
            c = contain(sid, b)
            if c < DRIFT_MAX_CONTAIN:
                add("bbox_drift", 1 - c, f"chỉ {c:.0%} pixel của segment nằm trong box (IoU bbox {iou:.2f})", b, sid)

    # missing_box: segment thing đủ lớn mà không box cùng lớp nào chứa phần lớn pixel của nó.
    for s in segs:
        sid = s["id"]
        if sid in used_s or seg_px[sid] < MISSING_MIN_FRAC * W * H:
            continue
        cover = max((contain(sid, b) for b in boxes if b["cls"] == seg_name[sid]), default=0.0)
        if cover < MISSING_MAX_COVER:
            add("missing_box", seg_px[sid] / (W * H), f"segment {seg_name[sid]} {seg_px[sid]} px "
                f"({seg_px[sid] / (W * H):.1%} ảnh) không có box cùng lớp (phủ tối đa {cover:.0%})", None, sid)
    return out


def cross_check(dv: Delivery, skip: set[str]) -> list[dict]:
    """Ứng viên lỗi gán nhãn cho cả lô. Bỏ qua ảnh đã có lỗi luật (mỗi ảnh tối đa một lỗi)."""
    out = []
    for i, name in enumerate(sorted(dv.manifest), start=1):
        if name not in skip:
            out += cross_check_image(dv, name)
        if i % 50 == 0:
            print(f"  đối chiếu {i}/{len(dv.manifest)} ảnh", file=sys.stderr, flush=True)
    out.sort(key=lambda c: (-c["score"], c["image"]))
    return out


# --------------------------------------------------------------------------- gộp kết quả
def merge(*per_image: dict[str, set[str]]) -> dict[str, str]:
    """Gộp nhiều dict {ảnh: {lỗi}} và giữ một lỗi mỗi ảnh theo PRIORITY."""
    allk: dict[str, set[str]] = {}
    for part in per_image:
        for name, kinds in part.items():
            allk.setdefault(name, set()).update(kinds)
    out = {}
    for name, kinds in allk.items():
        best = min(kinds, key=PRIORITY.index)
        out[name] = best
        if len(kinds) > 1:
            print(f"[ghi chú] {name} dính {sorted(kinds)} -> giữ {best}", file=sys.stderr)
    return out


def run_rules(dv: Delivery) -> list[dict]:
    per_image = merge(
        check_files(dv),
        check_guideline(dv),
        check_detection(dv),
        check_panoptic(dv),
    )
    errors = [{"type": kind, "target": name} for name, kind in sorted(per_image.items())]
    errors += [{"type": "scene_leak", "target": scene} for scene in check_scene_leak(dv)]
    return errors


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validator Nhóm 10")
    ap.add_argument("delivery", help="thư mục delivery/")
    ap.add_argument("--out", required=True, help="errors.json")
    ap.add_argument("--candidates", help="candidates.json (Pha 2)")
    ap.add_argument("--rules-out", help="ghi riêng lỗi luật (để merge_review.py gộp với review_log)")
    args = ap.parse_args()

    dv = load_delivery(Path(args.delivery))
    errors = run_rules(dv)
    write_json(Path(args.out), {"errors": errors})
    if args.rules_out:
        write_json(Path(args.rules_out), {"errors": errors})
    counts = Counter(e["type"] for e in errors)
    print(f"{len(dv.manifest)} ảnh -> {len(errors)} lỗi luật: {dict(counts)}")
    if args.candidates:
        cands = cross_check(dv, skip={e["target"] for e in errors})
        write_json(Path(args.candidates), {"candidates": cands})
        print(f"{len(cands)} ứng viên trên {len({c['image'] for c in cands})} ảnh: "
              f"{dict(Counter(c['type'] for c in cands))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
