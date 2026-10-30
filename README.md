# QI-RAG: Question-Indexed Retrieval-Augmented Generation

> **Research status:** Manuscript under review (October 2026).  
> **Study:** *An Empirical Evaluation of Evidence Retrieval, Answer-Candidate Coverage, and Selective Answering in Question-Indexed RAG*  
> **Software licensing:** Pending institutional review. See [License](#license).

## Overview

QI-RAG is a retrieval-augmented generation (RAG) architecture that retrieves **indexed reference questions** and supplies their linked **original passages** to a language model. Unlike conventional passage-based RAG, QI-RAG uses question similarity as the first retrieval step.

This repository accompanies an empirical evaluation of the existing QI-RAG design pattern. **The study evaluates both its advantages and its limitations; it does not claim that QI-RAG consistently outperforms Vanilla RAG or reduces semantic hallucinations.**

## Retrieval Design

1. **Offline indexing:** Embed 90,447 existing HotpotQA training questions using `e5-base-v2` and retain mappings from each reference question to its linked original passages.
2. **Question retrieval:** Rank indexed reference questions against an input question using embedding inner-product similarity.
3. **Evidence construction:** Collect linked passages in retrieval order, remove duplicate passages, and apply a final passage budget (`P = 2` or `P = 4`).
4. **Answer generation:** Supply the selected passages and the input question to a language model, which may answer or abstain under the configured response policy.
5. **Evaluation:** Measure answer accuracy, response coverage, supporting-document retrieval, stored-answer candidate availability, and selective-answering behavior separately.

The baseline index uses existing reference questions; synthetic questions are introduced only in the index-expansion experiment.

## Architecture

![QI-RAG Architecture](docs/architecture.png)

## Processing Flow

![QI-RAG Processing Flow](docs/flowchart.png)

## Evaluation Setup

| Component | Configuration |
|---|---|
| Dataset | HotpotQA (distractor development questions for evaluation) |
| Reference-question index | 90,447 training questions |
| Embedding model | `e5-base-v2` |
| Answer-generation model, main evaluations | `Qwen2.5-7B-Instruct-AWQ` |
| Common system comparison | 2,000 questions; QI-RAG, Vanilla RAG, and a research-index RePAQ-style baseline |
| Frozen holdout | 4,405 questions; separate 1,000-question calibration split |
| Passage budgets | `P = 2` and `P = 4` for the common comparison; `P = 2` for frozen-holdout retrieval |
| Frozen-holdout generation settings | Temperature `0.0`; maximum output `128` tokens |
| Main answer metrics | All-question exact match (EM), token F1, and response coverage; abstentions score zero on EM/F1 |
| Retrieval metrics | Document-title Recall@2 and Both-Gold Hit (complete retrieval of all gold document titles) |

**Experimental-lineage note:** The auxiliary retriever/payload experiments (A–C′) constitute a separate execution lineage. Their settings and exact question-ID equivalence with the common 2,000-question comparison have not been fully established, and their scores should not be pooled with the main comparison.

## Main Results

### 1. End-to-End Answering: Common 2,000-Question Evaluation

| System | EM (%) | Token F1 (%) | Response coverage (%) |
|---|---:|---:|---:|
| RePAQ-style (research QA index) | 6.60 | 9.21 | 100.00 |
| QI-RAG (`P = 2`) | 13.25 | 17.04 | 23.80 |
| QI-RAG (`P = 4`) | 16.10 | 20.74 | 30.75 |
| Vanilla RAG (`P = 2`) | 21.90 | 27.56 | 41.75 |
| Vanilla RAG (`P = 4`) | 23.45 | 29.53 | 45.85 |

These are **end-to-end system comparisons**, not isolated causal tests of the retrieval architecture: index sizes, information payloads, generation requirements, and abstention behavior were not all matched. The RePAQ-style condition is a research-index baseline, **not** a reproduction of full-scale official RePAQ.

### 2. Supporting-Document Retrieval: Frozen 4,405-Question Holdout

| Metric | QI-RAG (%) | Vanilla RAG (%) |
|---|---:|---:|
| Overall Recall@2 | 27.91 | 43.47 |
| Bridge Recall@2 | 22.78 | 32.60 |
| Comparison Recall@2 | 48.25 | 86.58 |
| Bridge Both-Gold Hit | 14.18 | 11.48 |
| Comparison Both-Gold Hit | 1.35 | 74.75 |

QI-RAG had a **limited Bridge-specific advantage** in complete gold-document retrieval, but lower overall document-title Recall@2 and substantially lower complete retrieval on Comparison questions.

Under the STRICT abstention policy on the same holdout, QI-RAG achieved **EM 12.01%, F1 15.27%, and coverage 22.47%**, compared with **21.93%, 27.23%, and 40.30%** for Vanilla RAG.

### 3. Stored-Answer Candidate and Payload Audits

Among the 4,405 holdout questions, the reconstructed stored-answer selection pipeline produced the following mutually exclusive categories:

| Category | Questions | Share (%) |
|---|---:|---:|
| No definite matching answer among top-eight reference-question candidates | 3,521 | 79.93 |
| Uncertain candidate-match status | 7 | 0.16 |
| Matching candidate found but omitted by two-answer selection | 351 | 7.97 |
| Matching stored answer retained in reconstructed payload | 526 | 11.94 |

In a **separate 3,843-question information-availability audit**, matching reference-answer token sequences were found in **7.81%** of stored-answer payloads and **35.44%** of original-passage payloads (+27.63 percentage points). This measures token-sequence availability, not equivalent-budget answer accuracy.

The stored-answer funnel is based on a **reconstructed** two-answer payload; the original transmitted HTTP request payloads were not independently verified.

### 4. Index Expansion and Passage Selection

- Adding **25,000** and **50,000** synthetic-question entries did **not** improve QI-RAG Recall@2: **27.91% → 27.83% → 27.91%** across the baseline, +25K, and +50K indexes.
- An **exploratory, post-hoc** fixed-order passage-selection intervention changed the selected passages for 183 of 4,405 questions. Across the full holdout, it was associated with **+1.36 percentage points EM** (95% CI [+1.00, +1.75]) and **+1.51 percentage points F1** (95% CI [+1.13, +1.92]). The all-question wrong-answer rate also increased by **0.43 percentage points**.

### 5. Selective Answering and Question Variations

For equal numbers of selected non-abstaining responses, the QI-RAG minus Vanilla RAG **Answered-EM** differences were **+2.61**, **0.00**, and **−2.67 percentage points** at the 20%, 30%, and 40% response budgets, respectively. These exploratory results do not establish a consistent advantage; the systems did not necessarily select identical question IDs.

For **2,000 matched parent questions** under each question-variation condition:

| Question variant | QI-RAG token F1 | Vanilla RAG token F1 |
|---|---:|---:|
| Original | 0.1704 | 0.2756 |
| Paraphrase | 0.1507 | 0.2472 |
| Noisy | 0.1418 | 0.2165 |

Vanilla RAG achieved higher F1 in all three tested conditions. Although QI-RAG had a smaller absolute Original-to-Noisy F1 decline, the study does **not** establish superior noise robustness: baseline performance differed, and semantic equivalence of the constructed variations was not independently confirmed.

## Scope and Limitations

- Document-title Recall@2 and Both-Gold Hit do **not** verify the presence of exact supporting sentences.
- EM, token F1, and answered-only incorrect-answer risk do **not** directly measure semantic hallucination or evidence grounding.
- The passage-selection intervention and matched-budget selective-answering comparisons are **post-hoc/exploratory**, not independent confirmatory tests.
- Findings primarily concern HotpotQA and the evaluated configurations; broader domain or language generalization is unverified.
- Some auxiliary experiments have different or incompletely verified execution settings. See the manuscript's **Appendix A, Table A1** before comparing results across lineages.

## Implementation and Reproducibility

This repository provides research-oriented materials associated with QI-RAG. Reproducing a reported number requires matching the relevant **execution lineage**, data split, index, retrieval and passage budgets, generation model, prompt/abstention policy, scoring implementation, and random seeds. The paper's **Tables 1–20 and Appendix A (Table A1)** define the reported evaluation conditions and results.

The principal dataset is publicly available from [HotpotQA](https://hotpotqa.github.io/). The repository's available code and outputs should be checked against the corresponding manuscript experiment before treating a run as reproduced.

## Manuscript

**Lee, J.** *An Empirical Evaluation of Evidence Retrieval, Answer-Candidate Coverage, and Selective Answering in Question-Indexed RAG*. Manuscript submitted to *Applied Sciences*, October 2026.

No DOI or publication acceptance is claimed while the manuscript is under review.

## Funding

This research was supported by the Ministry of Culture, Sports and Tourism (MCST) and the Korea Creative Content Agency (KOCCA), Grant **RS-2025-25459094**.

## License

This repository accompanies a manuscript under review. As stated in the existing repository notice, the software's copyright is held by the **Gwangju Institute of Science and Technology (GIST)**. Licensing terms are pending institutional review.

**No open-source license is currently granted by this README.** Until an authorized `LICENSE` file is published, do not assume permission to redistribute, modify, or commercially reuse the source code. Requests for licensing or other permissions should be directed to the institutional rights holder.

## Contact

**Junhyeong Lee**  
Gwangju Institute of Science and Technology (GIST)  
**junh.lee@gist.ac.kr**
