# Bảo mật khi làm việc — lô lab250_cc2a8eefdd81 (seed 10007)

## 1. Ca USB: hết giờ, muốn chép lô về nhà làm tiếp
Không chép lô `lab250_cc2a8eefdd81` (250 ảnh + nhãn + mask, 139 ảnh có người) hay gói `lab250_cc2a8eefdd81_v2.zip` ra USB. USB dễ mất, không mã hoá, và không còn kiểm soát được ai đọc dữ liệu. Gửi về email cá nhân hay nhờ người ngoài nhóm làm hộ cũng sai vì dữ liệu rời khỏi hệ thống công ty: không còn log truy cập, không thu hồi được, và người nhận không được giao quyền. Cách đúng: dừng ở mốc hiện tại, commit phần việc (validator, review_log) vào repo nhóm, ghi lại còn thiếu gì; hôm sau làm tiếp trên máy công ty hoặc qua kênh truy cập từ xa được cấp (VPN/máy ảo), và báo trưởng nhóm nếu cần thêm thời gian thay vì tự mang dữ liệu đi.

## 2. Năm việc: được hay không
- **Mở job được giao trên máy công ty:** Được. Đúng người, đúng job (ví dụ J04 của A04/Q02), đúng máy được cấp; mọi truy cập có log.
- **Chụp màn hình gửi người quen:** Không. Ảnh chụp chứa dữ liệu thật (ở lô này có mặt người rõ), rời khỏi hệ thống và không thu hồi được; hỏi nghiệp vụ thì mô tả bằng lời hoặc hỏi trong kênh nội bộ.
- **Chép lô ra USB:** Không. Như mục 1: mất kiểm soát dữ liệu, không có log, không thu hồi được.
- **Dán nhãn thật lên chatbot:** Không với dữ liệu thật của dự án: nội dung có thể bị lưu hoặc dùng lại ngoài hợp đồng. Bài lab dùng ảnh COCO công khai nên được hỏi AI nhưng nhóm vẫn ghi lại trong `prompts.md`.
- **Đưa tài khoản CVAT cho đồng nghiệp:** Không. Tài khoản gắn với người: lineage và `jobs.csv` ghi ai gán ai kiểm; dùng chung thì truy vết (như ticket bottle phải báo A04, A05, Q01–Q03) sai người. Đồng nghiệp cần quyền thì xin cấp tài khoản riêng.

## 3. Ca khuôn mặt: thấy mặt người rõ trong lô, guideline yêu cầu làm mờ
Lô này có nhiều ảnh thấy rõ mặt người, ví dụ `000000004395` (nam mặc sơ mi, cà vạt, mặt nhìn thẳng), `000000001296` (cô gái cầm điện thoại, mặt rõ) và `000000016598` (người tự chụp qua gương, mặt rõ). Cả ba đều nằm trong v2. Nếu guideline yêu cầu làm mờ thì lô ở trạng thái **Hold**: không giao tiếp, không train, và **báo người phụ trách** (trưởng dự án / người phụ trách dữ liệu) kèm danh sách ảnh. Không tự làm mờ hay xoá ảnh vì: (1) sửa ảnh sẽ đổi sha256, phá manifest và lineage, giống ghi đè v1; (2) làm mờ có thể làm sai nhãn (box person, mask) mà không ai kiểm lại; (3) quyết định xử lý dữ liệu cá nhân (làm mờ, xoá, xin đồng ý) thuộc người phụ trách và quy trình pháp lý, không phải người gán nhãn. Sau khi có hướng xử lý, ảnh sửa phải ra thành một lô mới có parent và manifest mới.
