"""Test validator Nhóm 10 trên một lô giả nhỏ dựng trong thư mục tạm (không cần dữ liệu COCO).

  py -m unittest tests.test_validator -v
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validator"))
import validate  # noqa: E402
import pngutil  # noqa: E402  (validate đã thêm scripts/ vào sys.path)

W, H = 100, 50
SEG_ID = 1 + 2 * 256          # segment thing: id 513, màu (1, 2, 0)
SEG_AREA = 20 * 10            # vùng x 40-59, y 20-29


def write_mask(path: Path) -> None:
    px = bytearray(W * H * 3)
    r, g, b = pngutil.segment_rgb(SEG_ID)
    for y in range(20, 30):
        for x in range(40, 60):
            i = (y * W + x) * 3
            px[i:i + 3] = bytes((r, g, b))
    pngutil.write_rgb(path, W, H, bytes(px))


def build(tmp: Path, *, missing=(), fake_hash=(), bad_guideline=(), leak_scene=None,
          labels=None, seg_override=None) -> Path:
    """4 ảnh 100x50, 2 cảnh (s1: a,b; s2: c,d), 1 job v1.2.
    labels: {ảnh: nội dung file YOLO}; seg_override: {ảnh: dict sửa segment}."""
    labels = labels or {}
    seg_override = seg_override or {}
    d = tmp / "delivery"
    for sub in ("images", "labels", "masks"):
        (d / sub).mkdir(parents=True)
    names = ["a", "b", "c", "d"]
    manifest, meta, imgs, anns = {}, [], [], []
    for i, n in enumerate(names):
        data = f"jpg-{n}".encode()
        if n not in missing:
            (d / "images" / f"{n}.jpg").write_bytes(data)
        (d / "labels" / f"{n}.txt").write_text(labels.get(n, "0 0.5 0.5 0.2 0.2\n"), encoding="utf-8")
        write_mask(d / "masks" / f"{n}.png")
        imgs.append({"id": i + 1, "file_name": f"{n}.jpg", "width": W, "height": H})
        seg = {"id": SEG_ID, "category_id": 1, "area": SEG_AREA, "bbox": [40, 20, 20, 10], "iscrowd": 0}
        seg.update(seg_override.get(n, {}))
        anns.append({"image_id": i + 1, "file_name": f"{n}.png", "segments_info": [seg]})
        h = hashlib.sha256(data).hexdigest()
        manifest[n] = {"image": f"images/{n}.jpg", "label": f"labels/{n}.txt", "mask": f"masks/{n}.png",
                       "image_sha256": ("0" * 64) if n in fake_hash else h}
        scene = "s1" if i < 2 else "s2"
        split = "train" if scene == "s1" else "val"
        if leak_scene == scene and i % 2:
            split = "train" if split == "val" else "val"
        meta.append({"image": n, "scene_id": scene, "split": split, "job_id": "J01",
                     "guideline_version": "v1.3" if n in bad_guideline else "v1.2"})
    (d / "manifest.json").write_text(json.dumps({"items": manifest}), encoding="utf-8")
    (d / "metadata.json").write_text(json.dumps({"items": meta}), encoding="utf-8")
    with (d / "jobs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "annotator", "reviewer", "guideline"])
        w.writeheader()
        w.writerow({"job_id": "J01", "annotator": "A1", "reviewer": "R1", "guideline": "v1.2"})
    (d / "classes.txt").write_text("0 person\n1 bicycle\n", encoding="utf-8")
    cats = [{"id": 1, "name": "person", "isthing": 1}, {"id": 2, "name": "bicycle", "isthing": 1}]
    (d / "panoptic.json").write_text(json.dumps({"categories": cats, "images": imgs, "annotations": anns}), encoding="utf-8")
    return d


def errors_of(d: Path) -> set[tuple[str, str]]:
    return {(e["type"], e["target"]) for e in validate.run_rules(validate.load_delivery(d))}


class TestRules11(unittest.TestCase):
    def test_clean_batch_reports_nothing(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t))), set())

    def test_missing_image(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t), missing={"b"})), {("missing_image", "b")})

    def test_wrong_hash(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t), fake_hash={"c"})), {("wrong_hash", "c")})

    def test_guideline_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t), bad_guideline={"d"})), {("guideline_mismatch", "d")})

    def test_scene_leak_targets_scene(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t), leak_scene="s2")), {("scene_leak", "s2")})

    def test_one_error_per_image_by_priority(self):
        with tempfile.TemporaryDirectory() as t:
            got = errors_of(build(Path(t), fake_hash={"a"}, bad_guideline={"a"}))
            self.assertEqual(got, {("wrong_hash", "a")})


class TestDetection12(unittest.TestCase):
    def run_labels(self, text):
        with tempfile.TemporaryDirectory() as t:
            return errors_of(build(Path(t), labels={"a": text}))

    def test_invalid_class(self):
        self.assertEqual(self.run_labels("80 0.5 0.5 0.2 0.2\n"), {("invalid_class", "a")})
        self.assertEqual(self.run_labels("-1 0.5 0.5 0.2 0.2\n"), {("invalid_class", "a")})
        self.assertEqual(self.run_labels("79 0.5 0.5 0.2 0.2\n"), set())

    def test_bbox_slightly_out_of_bounds(self):
        # tràn mép phải 0,5% (0,5 px trên ảnh rộng 100) vẫn phải bắt
        self.assertEqual(self.run_labels("0 0.905 0.5 0.2 0.2\n"), {("bbox_out_of_bounds", "a")})
        # chạm đúng mép (sai số làm tròn 6 chữ số) thì không báo
        self.assertEqual(self.run_labels("0 0.900000 0.5 0.200001 0.2\n"), set())

    def test_duplicate_box_same_size_shift_2px(self):
        text = "0 0.50 0.5 0.2 0.2\n0 0.52 0.5 0.2 0.2\n"   # lệch 2 px theo x
        self.assertEqual(self.run_labels(text), {("duplicate_box", "a")})

    def test_natural_overlap_not_duplicate(self):
        # chồng nhau nhiều nhưng khác kích thước -> không phải box lặp
        self.assertEqual(self.run_labels("0 0.50 0.5 0.20 0.2\n0 0.51 0.5 0.24 0.2\n"), set())
        # cùng kích thước, cùng vị trí nhưng khác lớp -> không phải box lặp
        self.assertEqual(self.run_labels("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n"), set())


class TestPanoptic13(unittest.TestCase):
    def test_invalid_category_id(self):
        with tempfile.TemporaryDirectory() as t:
            got = errors_of(build(Path(t), seg_override={"b": {"category_id": 999}}))
        self.assertEqual(got, {("invalid_category_id", "b")})

    def test_thing_overlap_area_too_big(self):
        with tempfile.TemporaryDirectory() as t:
            got = errors_of(build(Path(t), seg_override={"c": {"area": SEG_AREA + 40}}))
        self.assertEqual(got, {("thing_overlap", "c")})

    def test_area_exact_is_clean(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(errors_of(build(Path(t))), set())


class TestCrossCheck21(unittest.TestCase):
    """Segment person nằm ở x 40-59, y 20-29; box '0 0.5 0.5 0.2 0.2' trùng khít segment."""

    def cands(self, text):
        with tempfile.TemporaryDirectory() as t:
            dv = validate.load_delivery(build(Path(t), labels={"a": text}))
            return {(c["image"], c["type"]) for c in validate.cross_check(dv, skip=set())}

    def test_clean_pair_no_candidate(self):
        self.assertEqual(self.cands("0 0.5 0.5 0.2 0.2\n"), set())

    def test_swap(self):
        self.assertEqual(self.cands("1 0.5 0.5 0.2 0.2\n"), {("a", "class_or_category_swap")})

    def test_drift(self):
        self.assertEqual(self.cands("0 0.56 0.5 0.2 0.2\n"), {("a", "bbox_drift")})

    def test_missing_box(self):
        self.assertEqual(self.cands(""), {("a", "missing_box")})

    def test_merge(self):
        # hai box person, mỗi box phủ nửa segment -> segment bị gộp
        self.assertEqual(self.cands("0 0.45 0.5 0.1 0.2\n0 0.55 0.5 0.1 0.2\n"), {("a", "instance_merge")})


if __name__ == "__main__":
    unittest.main()
