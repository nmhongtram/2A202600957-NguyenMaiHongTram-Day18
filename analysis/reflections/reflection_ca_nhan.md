# Individual Reflection - Lab 18

**Tên:** Cá nhân  
**Module phụ trách:** M1, M2, M3, M4, M5

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Nhóm câu theo similarity; fallback token-overlap giúp chạy offline khi chưa có model. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent giữ ngữ cảnh dài, child tăng độ chính xác truy xuất. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Header Markdown được giữ trong text và metadata `section`. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF gộp kết quả lexical và vector, giảm phụ thuộc một phương pháp tìm kiếm. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea` được dùng nếu có; fallback regex giúp test vẫn chạy. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Có thể bật model thật bằng `USE_RAG_MODELS=1`; offline dùng lexical scorer. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Chạy lại bằng OpenAI API thật; Production đạt faithfulness 0.7463, answer_relevancy 0.6644, context_precision 0.1899, context_recall 0.7095. |
| Failure diagnostic tree | M4 | `failure_analysis()` | Bottom failures chủ yếu rơi vào context_precision, gợi ý thêm rerank/filter. |
| Contextual embeddings | M5 | `contextual_prepend()` và `_enrich_single_call()` | Enrichment thêm context/source trước chunk để giảm vocabulary gap. |

## Phần 2: Khó khăn & Giải quyết

- **Lỗi gặp phải:** `ModuleNotFoundError: No module named 'pypdf'` khi `load_documents()` đọc PDF.
- **Cách debug:** Chạy `pytest tests/ -v`, xác định lỗi ở `_extract_pdf_text()`, sau đó thêm fallback bỏ qua PDF nếu thiếu `pypdf`.
- **Lỗi môi trường:** `check_lab.py` bị `UnicodeEncodeError` trên Windows console.
- **Cách debug:** Chạy lại bằng `python -X utf8 check_lab.py`.
- **Lỗi hiệu năng:** Test treo lâu khi `sentence_transformers` cố load/download model.
- **Cách giải quyết:** Thêm công tắc `USE_RAG_MODELS=1`; mặc định dùng fallback nhanh, khi production có thể bật model thật.
- **Kết quả run thật:** Production tăng context precision nhẹ nhưng giảm faithfulness/answer relevancy/context recall so với naive baseline.
- **Cách debug:** Đọc `reports/ragas_report.json`, lấy bottom failures và thấy lỗi chính là retrieval sai domain, multi-hop thiếu context, và một số câu LLM trả lời "Không tìm thấy" dù context có đáp án.
- **Bài học:** Enrichment và reranking không tự động tốt hơn baseline nếu chưa có metadata filter, version filter và parent retrieval phù hợp.

## Phần 3: Action Plan cho project

## Project: Production RAG nội bộ

### Hiện tại
- RAG pipeline hiện tại: load tài liệu Markdown/PDF text, chunk, enrich, hybrid search, rerank, trả lời và evaluate.
- Known issues: câu hỏi multi-hop, versioning, số học và phê duyệt theo ngưỡng vẫn dễ retrieve nhầm context; run thật cho thấy các lỗi này ảnh hưởng trực tiếp đến answer relevancy.

### Plan áp dụng
1. [ ] Chunking strategy: dùng hierarchical + structure-aware; parent return để không mất context.
2. [ ] Search: hybrid BM25 + dense; BM25 xử lý keyword/số tốt, dense xử lý diễn đạt tự nhiên.
3. [ ] Reranking: bật CrossEncoder `BAAI/bge-reranker-v2-m3` bằng `USE_RAG_MODELS=1`, sau đó benchmark top-k.
4. [ ] Evaluation: dùng RAGAS thật bằng `USE_OPENAI_API=1`, kết hợp bottom-5 failure analysis thủ công sau mỗi lần chạy.
5. [ ] Enrichment: dùng combined single-call để tạo summary, questions, context và metadata trong một lần gọi.
6. [ ] Metadata: thêm `category`, `effective_date`, `version_status`, `department` để filter trước rerank.

### Timeline
- Tuần 1: Hoàn thiện metadata extraction và version filter.
- Tuần 2: Bật model thật, benchmark latency BM25/dense/rerank.
- Tuần 3: Chạy RAGAS thật, phân tích bottom-10 failures.
- Tuần 4: Tối ưu prompt answer, thêm citation và guardrail tránh trả lời "Không tìm thấy" khi context có số liệu trực tiếp.

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Problem solving | 4 |
| Evaluation mindset | 4 |
