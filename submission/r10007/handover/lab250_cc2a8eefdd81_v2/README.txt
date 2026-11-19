Release: lab250_cc2a8eefdd81_v2
Parent: lab250_cc2a8eefdd81
Số ảnh: 237
Số nhãn: 1671
Guideline: v1.2 (J01-J04), v1.3 (J05-J10)
QC v1: 22/250 = 8,8%
Kết luận v1: Hold — 8,8% ảnh/cảnh lỗi > ngưỡng 2%
Kết luận v2: Release — v2 gồm ảnh + nhãn detection (không kèm mask panoptic); lỗi còn lại trong v2 0/237 = 0% < 2% ⇒ được dùng để train. 22 lỗi của v1 = 13 ảnh lỗi detection/thiếu file tách sang rework_queue + 3 lỗi đã sửa trong v2 (1 hash, 1 guideline, 1 cảnh rò split) + 6 lỗi chỉ nằm ở mask panoptic của v1
Ghi chú:
- Không sửa v1 (lab250_cc2a8eefdd81); v2 chép nguyên bytes ảnh và nhãn của v1, không sửa tay nhãn nào.
- Rework 13 ảnh (rời v2): 000000001490 missing_image; 000000003156 missing_box; 000000005586 bbox_out_of_bounds; 000000006954 bbox_out_of_bounds; 000000009772 bbox_drift; 000000013597 missing_image; 000000014888 class_swap; 000000015660 invalid_class; 000000017905 bbox_drift; 000000019402 missing_box; 000000021879 duplicate_box; 000000022192 bbox_drift; 000000022892 invalid_class.
- Giao lại theo jobs.csv: A01/Q03: 3 ảnh; A02/Q01: 3 ảnh; A02/Q03: 2 ảnh; A03/Q02: 1 ảnh; A04/Q02: 1 ảnh; A05/Q01: 1 ảnh; A05/Q03: 2 ảnh.
- Hash: 000000022479 đã mở ảnh kiểm (mở được, 640x605 RGB), manifest v1 ghi sai -> manifest.csv ghi sha256 thật.
- Guideline: truyvet.csv lấy theo job trong jobs.csv; ảnh ghi sai guideline ở metadata: 000000021503.
- Split: scene_322 {'train': 4, 'val': 1} -> toàn bộ train (chuyển 000000006213).
- Lỗi chỉ ở panoptic, nhãn detection vẫn đúng nên ảnh ở lại v2, đã báo team panoptic: 000000002153 thing_overlap; 000000009769 category_swap; 000000013923 thing_overlap; 000000015440 instance_merge; 000000016249 category_swap; 000000021167 invalid_category_id.
