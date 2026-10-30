# QI-RAG: Question-Indexed Retrieval-Augmented Generation

> **Status:** Manuscript under review (October 2026)  
> **Manuscript:** *An Empirical Evaluation of Evidence Retrieval, Answer-Candidate Coverage, and Selective Answering in Question-Indexed RAG*  
> **Author:** Junhyeong Lee — Gwangju Institute of Science and Technology (GIST)  
> **Licensing:** Pending institutional review; see [License](#license).

## Overview

**QI-RAG** is a question-indexed retrieval-augmented generation framework. It retrieves **similar reference questions** and supplies their linked **original passages** to a large language model (LLM) for answering or abstention.

The central distinction is the **information returned after question retrieval**:

- **RePAQ-style question–answer retrieval:** retrieves similar reference questions and directly returns **stored answers**.
- **QI-RAG question-to-passage retrieval:** retrieves similar reference questions but passes **linked original passages** to an LLM for answer generation.
- **Vanilla RAG:** retrieves **passages directly** from the input question and passes them to an LLM.

This repository accompanies an **empirical evaluation of an existing QI-RAG design pattern**, not a claim of a newly invented retrieval method or universal superiority. The study compares question-indexed passage delivery with a **research-index RePAQ-style baseline** and with **Vanilla RAG**, while separately investigating evidence retrieval, candidate availability, passage selection, and selective answering.

**Main finding:** QI-RAG achieved higher exact match (EM) than the evaluated research-index RePAQ-style baseline in the common 2,000-question comparison, but with substantially lower response coverage. Against Vanilla RAG on a separate frozen 4,405-question holdout, QI-RAG had lower overall retrieval and answer accuracy, despite a localized Bridge-type complete-document retrieval advantage. These comparisons do **not** establish that QI-RAG generally improves accuracy, noise robustness, or semantic hallucination.

## 1. Retrieval Approaches: RePAQ-Style vs. QI-RAG vs. Vanilla RAG

| Property | RePAQ-style baseline | QI-RAG | Vanilla RAG |
|---|---|---|---|
| Primary retrieval unit | Reference question | Reference question | Passage |
| Retrieved information delivered to answer stage | Stored answer | Linked original passages | Directly retrieved passages |
| Final answering route in the common comparison | Direct stored-answer return | LLM generation / abstention | LLM generation / abstention |
| Central evaluation question | Is a matching stored answer retrieved? | Do similar questions lead to sufficient evidence? | Can direct passage retrieval obtain the needed evidence? |

**Important scope distinction:** “RePAQ-style” refers here to a **research question–answer index** used in the reported experiments. It is **not** a reproduction of the complete official PAQ/RePAQ-scale system. The question-oriented designs share a retrieval principle, but differences in index size, available information, generation requirements, and abstention policy prevent attribution of end-to-end score differences to payload choice alone.

### QI-RAG processing pipeline

1. **Offline indexing:** Embed 90,447 existing HotpotQA training questions with `e5-base-v2` and retain each question's links to its original passages.
2. **Question retrieval:** Rank indexed reference questions by inner-product similarity to the input-question embedding.
3. **Passage construction:** Collect associated passages in retrieval order, remove duplicates, and limit the final context to a passage budget `P`.
4. **Generation and abstention:** Supply the input question and selected passages to the LLM, which responds under the specified policy.
5. **Evaluation:** Score answers, abstentions, supporting-document retrieval, and candidate-selection outcomes separately.

The **baseline** question index uses existing reference questions. Synthetic questions were added **only in a separate index-expansion intervention**.

## 2. Architecture and Flow

![QI-RAG Architecture](docs/architecture.png)

![QI-RAG Processing Flow](docs/flowchart.png)

The two figures above use the existing repository paths. They should be checked against the final manuscript architecture before being treated as identical to its Figure 1.

## 3. Evaluation Setup

| Component | Reported configuration |
|---|---|
| Evaluation data | HotpotQA distractor development questions |
| Baseline reference-question index | 90,447 training questions |
| Embedding model | `e5-base-v2` |
| Main LLM generator | `Qwen2.5-7B-Instruct-AWQ` for QI-RAG and Vanilla RAG |
| Common system evaluation | 2,000 questions; RePAQ-style, QI-RAG, and Vanilla RAG |
| Frozen holdout | 4,405 questions; disjoint 1,000-question calibration split |
| Passage budgets | `P = 2` / `P = 4` in common comparison; `P = 2` for holdout retrieval |
| Holdout generation | Temperature `0.0`; maximum output `128` tokens |
| Answer metrics | All-question exact match (EM), token F1, response coverage |
| Retrieval metrics | Document-title Recall@2; Both-Gold Hit (all required document titles retrieved) |
| Abstention scoring | Abstentions contribute zero to all-question EM and F1 |

**Execution lineages matter.** The common 2,000-question evaluation, frozen holdout, 3,843-question payload audit, and **auxiliary A–C′** retriever/payload comparisons have distinct evaluation scopes. In particular, the auxiliary A–C′ experiment's exact question-ID equivalence and configuration equivalence with the common comparison have **not** been established. Do not pool those results or treat them as a controlled common benchmark. See manuscript **Appendix A, Table A1**.

## 4. Primary Comparison: QI-RAG and RePAQ-Style

### 4.1. End-to-end answering on 2,000 common questions

| System | EM (%) | Token F1 (%) | Response coverage (%) |
|---|---:|---:|---:|
| **RePAQ-style** (research QA index; direct answer) | **6.60** | **9.21** | **100.00** |
| **QI-RAG (`P = 2`)** | **13.25** | **17.04** | **23.80** |
| **QI-RAG (`P = 4`)** | **16.10** | **20.74** | **30.75** |

Relative to the evaluated RePAQ-style baseline, QI-RAG's observed EM was **+6.65 percentage points** at `P = 2` and **+9.50 percentage points** at `P = 4`. However, response coverage was **76.20** and **69.25 percentage points lower**, respectively. All-question EM/F1 include zero scores for abstentions.

**Interpretation:** These are **end-to-end differences between non-equivalent systems**. The result supports reporting QI-RAG's observed higher EM in this specific research-index comparison; it does **not** isolate the causal effect of substituting passages for stored answers, nor does it establish superiority over official full-scale RePAQ.

For completeness, the same common-question evaluation also reported **Vanilla RAG** at `P = 2`: EM **21.90%**, F1 **27.56%**, coverage **41.75%**; and at `P = 4`: EM **23.45%**, F1 **29.53%**, coverage **45.85%** (manuscript Table 9).

### 4.2. Stored-answer payload vs. original-passage payload

A **separate 3,843-question information-availability audit** compared normalized reference-answer token-sequence presence in reconstructed answer and passage payloads.

| Information payload | Questions | Reference-answer token-sequence presence (%) |
|---|---:|---:|
| QA: stored-answer payload | 3,843 | 7.81 |
| QC: linked original-passage payload | 3,843 | 35.44 |
| **QC − QA** | **3,843** | **+27.63 percentage points** |

Original-passage payloads more often contained a matching reference-answer token sequence than stored-answer payloads under the audited configurations. **This is an information-availability finding, not a matched-token-budget generation-accuracy comparison, and not a direct test of the official RePAQ system.** (Manuscript Table 12.)

### 4.3. Where stored-answer candidates are lost

For the **4,405-question frozen holdout**, a separate audit examined stored answers attached to the **top eight** retrieved reference-question candidates. It then reconstructed a payload retaining up to **two unique nonempty answers** in retrieval order.

| Candidate-selection category | Questions | Share (%) |
|---|---:|---:|
| No definite matching stored answer in top eight | 3,521 | 79.93 |
| Uncertain candidate-match status | 7 | 0.16 |
| Matching answer found but omitted by two-answer selection | 351 | 7.97 |
| Matching answer retained in reconstructed QA payload | 526 | 11.94 |
| **Total** | **4,405** | **100.00** |

This distinguishes **candidate unavailability** from **loss during answer selection**. Matching was assessed with normalized exact answers and recorded mapping checks. The original transmitted HTTP request payloads were **not independently verified**; these categories describe a **reconstructed QA payload**, not confirmed production requests. (Manuscript Table 14.)

### 4.4. Auxiliary retriever/payload experiments — separate lineage

The manuscript also reports the following **auxiliary A–C′** conditions. These **must not be presented as a controlled replication** of Section 4.1:

| Auxiliary condition | Coverage (%) | EM (%) | F1 (%) | Answer containment (%) |
|---|---:|---:|---:|---:|
| A: RePAQ official Dense Retriever + research QA index | 100.0 | 8.6 | 12.6 | 10.2 |
| B: E5 + stored-answer payload | 100.0 | 11.4 | 16.5 | 13.2 |
| B′: E5 + stored-answer payload + LLM | 9.3 | 3.0 | 4.0 | 3.4 |
| C: QI-RAG + E5 + passages, strict | 30.1 | 15.9 | 20.6 | 18.0 |
| C′: QI-RAG + E5 + passages, permissive | 37.2 | 19.0 | 24.1 | 21.3 |

The retriever, answer payload, generation requirements, and response policy are **not independently controlled** across these conditions; question-ID equivalence with the common 2,000-question evaluation also remains unverified. A and B include a retrospective **Oracle@2** assessment. These descriptive outcomes must remain associated with **Table 11 and Appendix A, Table A1** of the manuscript.

## 5. Additional Benchmark: QI-RAG vs. Vanilla RAG

### 5.1. Supporting-document retrieval on the frozen 4,405-question holdout

| Document-title metric | QI-RAG (%) | Vanilla RAG (%) |
|---|---:|---:|
| Overall Recall@2 | 27.91 | 43.47 |
| Bridge Recall@2 | 22.78 | 32.60 |
| Comparison Recall@2 | 48.25 | 86.58 |
| Bridge Both-Gold Hit | 14.18 | 11.48 |
| Comparison Both-Gold Hit | 1.35 | 74.75 |

QI-RAG showed a **localized numerical advantage** in complete gold-document retrieval for Bridge questions (**14.18% vs. 11.48%**), but lower overall Recall@2 and markedly lower complete retrieval on Comparison questions (**1.35% vs. 74.75%**). Recall@2 and Both-Gold Hit use document titles, **not verification of individual supporting sentences**. (Manuscript Table 13.)

Under the **STRICT abstention policy** on the same holdout:

| All-question metric | QI-RAG (%) | Vanilla RAG (%) |
|---|---:|---:|
| EM | 12.01 | 21.93 |
| Token F1 | 15.27 | 27.23 |
| Response coverage | 22.47 | 40.30 |

These data do **not** show a general end-to-end performance advantage for QI-RAG over Vanilla RAG. (Manuscript Table 10.)

### 5.2. Index expansion and passage selection

- **Index expansion:** Adding 25,000 and 50,000 synthetic-question entries yielded QI-RAG Recall@2 of **27.91% → 27.83% → 27.91%** (baseline / +25K / +50K); no improvement over baseline was observed at +50K (Table 16).
- **Exploratory fixed-order passage-selection intervention:** Passage selections changed for **183/4,405** questions; **366** new responses were generated across two conditions. Full-holdout changes were **EM +1.36 percentage points** (95% CI **[+1.00, +1.75]**) and **F1 +1.51 percentage points** (95% CI **[+1.13, +1.92]**); the all-question wrong-answer rate also rose **+0.43 percentage points** (Table 17). This was **post-hoc**, not confirmatory.

### 5.3. Selective answering and question variations

At response-selection budgets of 20%, 30%, and 40%, the **QI-RAG minus Vanilla RAG Answered-EM** differences were **+2.61, 0.00, and −2.67 percentage points**, respectively. The respective 95% intervals were **[−1.36, +6.92]**, **[−3.18, +3.25]**, and **[−5.22, 0.00]** percentage points. The selected question IDs were not necessarily identical between systems, and the analyses were exploratory (Table 18).

Across **2,000 matched parent questions per variant**:

| Variant | QI-RAG token F1 | Vanilla RAG token F1 |
|---|---:|---:|
| Original | 0.1704 | 0.2756 |
| Paraphrase | 0.1507 | 0.2472 |
| Noisy | 0.1418 | 0.2165 |

Vanilla RAG had higher F1 in every tested variant. The smaller **absolute** Original-to-Noisy F1 drop for QI-RAG (**−0.0286 vs. −0.0590**) does **not** establish superior robustness: baseline performance differed, and the meaning-preserving equivalence of paraphrases was not independently verified (Tables 19–20).

## 6. Scope, Limitations, and Reproducibility

- **System comparisons are not component-isolation experiments.** RePAQ-style, QI-RAG, and Vanilla RAG differ in available knowledge, answer generation, and response policies.
- **Do not conflate lineages.** The auxiliary A–C′ experiments are separate from the main common 2,000-question comparison; the 3,843-question audit and frozen 4,405-question holdout also have distinct scopes.
- **Retrieval metrics are document-title based.** They do not confirm that all required supporting sentences were present.
- **Answer metrics are not semantic hallucination metrics.** EM, token F1, and `1 − Answered-EM` do not directly assess evidence grounding or semantic hallucination.
- **Post-hoc analyses remain exploratory.** Passage selection and selective-answering analyses were developed after examining the holdout.
- **Generalization is unverified.** Primary results concern HotpotQA and the reported configurations; paraphrase semantic equivalence was not independently verified.

A valid reproduction requires alignment of **experiment lineage**, question IDs and data split, index/corpus, retrieval settings, passage budget, LLM and generation parameters, prompts, abstention rules, answer scorer, and random seeds. **Manuscript Tables 1–20 and Appendix A (Table A1)** provide the authoritative experimental descriptions and results. Do not claim that every result can be reproduced from this repository without first verifying that the corresponding scripts, inputs, and output artifacts are actually present.

- **Dataset:** [HotpotQA](https://hotpotqa.github.io/)
- **Repository:** [ljh77/QIRAG](https://github.com/ljh77/QIRAG)

## 7. Manuscript

**Lee, J.** *An Empirical Evaluation of Evidence Retrieval, Answer-Candidate Coverage, and Selective Answering in Question-Indexed RAG.* Manuscript submitted to *Applied Sciences*, October 2026 (**under review**).

No acceptance, publication, or DOI is claimed here.

## 8. Funding

This study was supported by the **Ministry of Culture, Sports and Tourism (MCST)** and the **Korea Creative Content Agency (KOCCA)** under Grant **RS-2025-25459094**.

## License

This repository accompanies a manuscript under review. Under the repository's existing rights notice, copyright in the software is held by the **Gwangju Institute of Science and Technology (GIST)**. Licensing terms are pending institutional review.

**This README does not grant an open-source license.** Unless and until an authorized license is published, do not assume permission to redistribute, modify, or commercially reuse the code. Direct licensing and permission requests to the institutional rights holder.

## Contact

**Junhyeong Lee**  
Gwangju Institute of Science and Technology (GIST)  
**junh.lee@gist.ac.kr**
