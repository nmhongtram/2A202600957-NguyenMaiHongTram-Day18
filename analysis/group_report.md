# Group Report - Lab 18: Production RAG

**Nhóm:** Cá nhân  
**Ngày:** 22/06/2026  
**Run:** OpenAI API key thật

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|------------|------------|
| Cá nhân | M1: Chunking | Done | 13/13 |
| Cá nhân | M2: Hybrid Search | Done | 5/5 |
| Cá nhân | M3: Reranking | Done | 5/5 |
| Cá nhân | M4: Evaluation | Done | 4/4 |
| Cá nhân | M5: Enrichment | Done | 10/10 |

Tổng kiểm thử: `37 passed`.

## Kết quả RAGAS

| Metric | Naive | Production | Delta |
|--------|-------|------------|-------|
| Faithfulness | 0.8628 | 0.7463 | -0.1165 |
| Answer Relevancy | 0.7463 | 0.6644 | -0.0819 |
| Context Precision | 0.1748 | 0.1899 | +0.0151 |
| Context Recall | 0.7866 | 0.7095 | -0.0771 |

## Key Findings

1. **Biggest improvement:** Production tăng nhẹ `context_precision` (+0.0151), cho thấy hybrid retrieval/enrichment giúp giảm một phần nhiễu context.
2. **Biggest challenge:** Các câu hỏi multi-hop như lương thử việc Junior, laptop 30 triệu, hoặc mua thiết bị 55 triệu vẫn thất bại do cần ghép nhiều tài liệu.
3. **Surprise finding:** Naive baseline có faithfulness và answer relevancy cao hơn production trong run thật, vì production enrichment/reranking đôi khi đưa context đúng xuống dưới hoặc làm LLM abstain sai.

## Presentation Notes (5 phút)

1. RAGAS scores: Production precision tăng nhẹ nhưng recall/relevancy giảm, cần tối ưu retrieval chứ không chỉ thêm LLM enrichment.
2. Biggest win: M2 hybrid + M5 enrichment giúp context precision nhích lên.
3. Case study: Câu "thiết bị 55 triệu" bị kéo nhầm các context "phê duyệt" thuộc domain nghỉ phép/tạm ứng.
4. Next optimization: metadata filter theo domain, parent retrieval, version filter, query decomposition cho multi-hop.
