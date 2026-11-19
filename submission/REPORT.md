# Báo cáo nhóm 10

> Mọi số dưới đây lấy từ lần chạy thật (`score_findings.py` in F1 làm tròn một chữ số; script đo ngưỡng (không nộp kèm repo)
> chỉ đọc `delivery/`). Không dùng `_private/`, `--dev-truth` hay `--truth-dir`.

## 1. Số liệu luyện tập
- F1 theo seed (rule / cross_check / overall), **trước review** (errors.json mới có lỗi luật):
  - seed 10007 (seed nhóm): ≈ 1.0 / — / ≈ 0.7
  - seed 11 (phụ, `inject_release.py`): ≈ 1.0 / — / ≈ 0.8
  - seed 22 (phụ): ≈ 1.0 / — / ≈ 0.7
  - **Sau review seed 10007: ≈ 1.0 / ≈ 0.8 / ≈ 0.9** (22 lỗi: 13 lỗi luật + 9 lỗi gán nhãn confirmed; truy vết ≈ 1.0).
    Chấm một lần sau khi đã chốt review; không sửa kết luận theo điểm.
- Lỗi luật trên lô sạch (`inject_release.py --seed 5 --no-faults`, 50 ảnh calib): **0**.
- Số ứng viên trên lô sạch: **0 / 50 ảnh** (bản validator đầu tiên: 2, đều là `instance_merge` do box lồng nhau).
- Hàng đợi trên seed luyện tập: **9 ảnh** ở seed 10007 (3 swap, 3 bbox_drift, 2 missing_box, 1 instance_merge);
  seed 11: 4; seed 22: 8.
- Ngưỡng đã chọn và cách chọn (không dùng `_private`):
  - Ghép box ↔ segment thing (bỏ `iscrowd`) qua **tên lớp**, tham lam 1-1 theo IoU bbox (+0,25 nếu cùng lớp).
  - Tín hiệu chính là **pixel mask**, không chỉ IoU: contain = tỉ lệ pixel của segment nằm trong box.
  - Chạy script đo ngưỡng (không nộp kèm repo) với ngưỡng rất lỏng trên lô sạch và 3 seed, xem phân bố điểm:
    ở lô sạch mọi cặp cùng lớp có contain ≥ 0,995, không có cặp khác lớp trùng vị trí, không có segment phủ hai box.
    Ứng viên drift ở các seed đều có contain ≤ 0,83 → khoảng trống 0,83–0,995.
  - Chọn: drift khi contain < 0,95; swap khi khác lớp và IoU ≥ 0,4; missing_box khi segment ≥ 0,2% ảnh và không box
    cùng lớp nào chứa > 50% pixel; instance_merge khi một segment chiếm ưu thế trong hai box cùng lớp và mỗi box giữ
    riêng ≥ 8% pixel. Đặt lệch về phía recall vì đề ẩn có thể lệch nhẹ hơn; hàng đợi vẫn nhỏ (≤ 9 ảnh / 250)
    nên xem hết được, người xem loại ca tự nhiên.
  - Một ứng viên tự nhiên lặp lại ở cả 3 seed (segment chiếm 44% ảnh, độ phủ đúng 0,5) nằm ngay mép ngưỡng 0,5 và
    không vào hàng đợi.

## 2. Vì sao báo hết ứng viên làm F1 giảm?
F1 = 2PR/(P+R) với precision P = TP/(TP+FP). Ứng viên chỉ là chỗ hai nguồn nhãn lệch nhau; COCO vốn lệch tự nhiên
(vật bị che, crowd, box lồng nhau) nên nhiều ứng viên không phải lỗi. Báo hết thì mỗi ca lệch tự nhiên thành một FP:
recall không tăng thêm (lỗi thật đã nằm trong danh sách) nhưng P giảm, nên F1 giảm. Ngoài ra ứng viên swap không nói
bên nào sai: đoán sai `class_swap`/`category_swap` vừa là FP vừa là FN.

Số của nhóm: bản validator đầu tiên (ghép thô) ra **33 ứng viên** trên seed 10007, trong đó 25 `instance_merge`, và
2 ứng viên trên lô sạch 50 ảnh, tức chừng 10 ca lệch tự nhiên / 250 ảnh. Bản đã hiệu chỉnh còn 9 ứng viên (là tập con
của 33 ứng viên đầu); sau khi xem ảnh cả 9 đều **confirmed**, 0 rejected. Nếu nộp cả 33 ứng viên thô mà không xem ảnh
thì ít nhất 24 mục `instance_merge` thừa là FP, precision phần gán nhãn tối đa 9/33 ≈ 0,27 và F1 phần này sụt mạnh,
trong khi recall không tăng. Ngoài ra 3 ca swap phải nhìn ảnh mới chọn được `class_swap` hay `category_swap`.

Hàng đợi bản cuối không có ca lệch tự nhiên nào (0 rejected) vì ngưỡng đã được đo trên lô sạch: với ngưỡng đã chọn,
lô sạch 50 ảnh cho 0 ứng viên, tức ngưỡng nằm ngoài vùng lệch tự nhiên của COCO (mục 1).

## 3. Quyết định và lineage
- QC v1: **22/250 = 8,8%** (13 lỗi luật + 9 lỗi gán nhãn; mỗi ảnh tối đa một lỗi, `scene_leak` tính một), lớn hơn
  ngưỡng 2% → **HOLD** (`lineage.py record --event decision --decision HOLD`). Chỉ riêng lỗi luật (13/250 = 5,2%) đã
  vượt ngưỡng, nên quyết định HOLD không phụ thuộc kết quả review.
- v2: 237 ảnh, 1671 nhãn; 13 ảnh lỗi nhãn detection / thiếu file sang `rework_queue.csv` (giao đúng annotator/reviewer
  theo `jobs.csv`); 6 ảnh chỉ lỗi panoptic ở lại v2 và được báo team panoptic trong README.
- `lineage.py impact` cho ra `R-021: affected=True action=review_or_retrain`: training run R-021 đã train trên v1
  `lab250_cc2a8eefdd81`. Model đã học từ nhãn sai (box thiếu/lệch/nhầm lớp, cảnh rò train/val làm điểm val ảo);
  sửa dataset thành v2 không thay đổi trọng số đã học, nên R-021 phải được đánh giá lại trên v2 sạch hơn và train
  lại nếu ảnh hưởng đáng kể. Lineage giữ liên kết R-021 → v1 → v2 để biết run nào cần làm lại.
- v2 `lab250_cc2a8eefdd81_v2` ghi event `release` với `--parent lab250_cc2a8eefdd81`.

## 4. Hash chứng minh gì, không chứng minh gì?
- **Chứng minh:** file trong v2 đúng từng byte với file lúc đóng gói (`manifest.csv` + `.sha256` của zip); v1 không bị
  sửa (H1 so release_id và hash nhãn/mask); nhãn trong v2 là nhãn gốc chưa sửa tay (H7). Ảnh `000000022479` có
  sha256 thật khác manifest v1 nhưng mở được, đúng nội dung → manifest v1 ghi sai, ghi lại hash thật trong v2.
- **Không chứng minh:** nhãn đúng. Hash chỉ nói "file không đổi", không nói box có đúng vật, lớp có đúng, có thiếu
  box hay không; một nhãn sai được hash vẫn là nhãn sai. Hash cũng không chứng minh ảnh không có dữ liệu cá nhân,
  hay split hợp lý. Việc đó cần validator + người xem ảnh.

## 5. Truy vết
- Changelog v1.3 đổi quy tắc lớp `bottle` (bị che > 50% hoặc cắt mép vẫn phải gán box; v1.2 cho phép bỏ qua).
  Ticket nêu `scene_195`, `scene_224`.
- Tiêu chí: ảnh thuộc **job guideline cũ** (v1.2: J01–J04 theo `jobs.csv`) và có ít nhất
  một **box detection** lớp `bottle` trong `labels/`.
  - tier_a = ảnh như vậy trong 2 cảnh ticket nêu: 2 ảnh.
  - tier_b = ảnh như vậy ở cảnh khác, chưa ai báo: 4 ảnh.
  - notify = annotator + reviewer của các job chứa ảnh tầng a, b (J02, J03, J04): A04, A05, Q01, Q02, Q03.
- **Vì sao lấy guideline theo job:** người gán làm theo bản guideline ghi ở đầu job; `guideline_version` của từng
  ảnh trong metadata có thể chính là lỗi `guideline_mismatch` (ở lô này ảnh `000000021503` ghi sai). Lấy theo ảnh sẽ
  bỏ sót hoặc kéo nhầm ảnh vào truy vết.
- Tự đo: truy vết F1 ≈ 1.0 / 1.0 / 1.0 trên seed 10007, 11, 22 (mỗi seed một lớp khác nhau: bottle, wine glass,
  truck), nên công cụ không gắn cứng với một lô.

## 6. Bug hoặc điểm chưa rõ trong lab (nếu có)
- Không thấy bug trong `scripts/`.
- Tải dữ liệu: server COCO chỉ cho khoảng 65 KB/s mỗi kết nối, nên `download_public.py` (tải tuần tự hai file zip
  ~1,1 GB) mất nhiều giờ. Nhóm tải hai file zip annotation song song nhiều kết nối vào `data_source/cache/`, rồi chạy
  `download_public.py` (script nhận file đã có trong cache). Không sửa script nào.
