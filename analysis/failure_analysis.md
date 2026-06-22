# Failure Analysis - Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Thành viên:** Cá nhân phụ trách M1-M5  
**Run:** Đã chạy lại với OpenAI API key thật

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|--------|---------------|------------|-------|
| Faithfulness | 0.8628 | 0.7463 | -0.1165 |
| Answer Relevancy | 0.7463 | 0.6644 | -0.0819 |
| Context Precision | 0.1748 | 0.1899 | +0.0151 |
| Context Recall | 0.7866 | 0.7095 | -0.0771 |

## Nhận xét nhanh

Production pipeline cải thiện nhẹ `context_precision` nhờ enrichment + hybrid retrieval + reranking, nhưng giảm ở `faithfulness`, `answer_relevancy` và `context_recall`. Các failure lớn nhất tập trung ở câu hỏi multi-hop, câu hỏi cần bảng lương/mua sắm, và trường hợp versioning giữa chính sách cũ/mới.

## Bottom-5 Failures

### #1
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng; lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Không tìm thấy."
- **Worst metric:** Answer Relevancy = 0.0000
- **Error Tree:** Output sai -> Context thiếu bảng lương Junior -> Query cần multi-hop
- **Root cause:** Retrieval lấy được chính sách thử việc 85% nhưng chưa lấy đúng `bang_luong_2024.md`.
- **Suggested fix:** Query decomposition: `lương thử việc 85%` + `Junior cao nhất bảng lương`; thêm metadata category `salary`.

### #2
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Cần phê duyệt theo ngưỡng mua sắm và xác nhận cấu hình kỹ thuật từ phòng CNTT.
- **Got:** Answer không khớp đầy đủ với câu hỏi.
- **Worst metric:** Answer Relevancy = 0.0000
- **Error Tree:** Output thiếu -> Context chưa đủ hai điều kiện -> Query multi-hop
- **Root cause:** Câu hỏi có hai intent: ngưỡng phê duyệt mua sắm và điều kiện thiết bị CNTT.
- **Suggested fix:** Tách câu hỏi thành hai sub-query rồi hợp nhất context trước khi generate.

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Không tìm thấy."
- **Worst metric:** Answer Relevancy = 0.0000
- **Error Tree:** Output sai -> Context sai domain -> Rerank chưa lọc được từ "phê duyệt" chung
- **Root cause:** Retrieval nhầm sang nghỉ không lương/phê duyệt chi phí thay vì chính sách mua sắm.
- **Suggested fix:** Thêm metadata filter `category=procurement` và boost các chunk có số tiền gần 50.000.000 VNĐ.

### #4
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** Không, nhân viên thử việc không được nghỉ phép năm.
- **Got:** Answer/context bị nhiễu bởi các chính sách nghỉ phép khác.
- **Worst metric:** Context Precision = 0.1028
- **Error Tree:** Output không chắc -> Context nhiễu -> Keyword "nghỉ phép" quá rộng
- **Root cause:** BM25 và dense search kéo nhiều tài liệu nghỉ phép nhưng chưa ưu tiên `thu_viec.md`.
- **Suggested fix:** Boost query terms `thử việc`, thêm metadata `employment_status`, và rerank theo phrase match.

### #5
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Got:** "Không tìm thấy."
- **Worst metric:** Answer Relevancy = 0.0000
- **Error Tree:** Context có đáp án -> LLM abstain sai -> Prompt/generation issue
- **Root cause:** Context chứa đúng hạn mức nhưng answer generator vẫn trả lời "Không tìm thấy", có thể do context bị nhiễu chunk bảo hiểm xã hội/nghỉ ốm.
- **Suggested fix:** Đặt instruction ưu tiên trả lời khi bất kỳ context nào chứa số liệu trực tiếp; rerank top-1 theo PVI/hạn mức.

## Case Study

**Question chọn phân tích:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

**Error Tree walkthrough:**
1. Output đúng? Không, model trả lời "Không tìm thấy."
2. Context đúng? Không đủ; các context top đầu bị hút về những đoạn có từ "phê duyệt" nhưng sai domain.
3. Query rewrite OK? Chưa; query cần hiểu "thiết bị trị giá 55 triệu" thuộc quy trình mua sắm.
4. Fix ở bước: retrieval + metadata filtering trước rerank.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata `category` thủ công hoặc LLM-extracted cho `mua_sam`, `tam_ung`, `chi_phi`, `nghi_phep`.
- Thêm `effective_date` và `version_status` để tránh trả lời theo policy cũ.
- Dùng parent retrieval: retrieve child nhưng đưa parent section đầy đủ vào LLM.
- Thêm query decomposition cho câu hỏi multi-hop.
- Chỉnh prompt trả lời: nếu context có số liệu trực tiếp thì không được trả lời "Không tìm thấy."
