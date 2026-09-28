# QI-RAG: Query-Indexed Retrieval-Augmented Generation(2026.4)

> **Status:** manuscript under review.
> Licensing terms pending institutional review — see [License](#license).
>
> 
## Overview
QI-RAG is a retrieval framework that indexes queries instead of documents.
It is designed to improve robustness under noisy queries and reduce hallucination in large language models (LLMs).

## Key Idea
Unlike standard RAG, which retrieves documents directly, QI-RAG:
- Matches input queries to pre-indexed queries
- Uses pre-mapped document sets
- Controls retrieval structure explicitly

## Architecture
![Architecture](docs/architecture.png)

## Flow
![Flow](docs/flowchart.png)

## Features
- Query-indexed retrieval
- Robust to noisy queries
- Reduced hallucination via constrained context
- Structured retrieval pipeline

## Implementation Note
This repository provides a simplified implementation for research purposes.

## Results
QI-RAG demonstrates improved robustness compared to standard RAG
under noisy and adversarial query settings.

## License

This repository accompanies a manuscript under review. The software was
developed under Grant RS-2025-25459094 (MCST/KOCCA); copyright is held by
the Gwangju Institute of Science and Technology. Licensing terms are being
finalized; until a license file is added, the code is provided for review
and reproduction of the reported experiments.

## Contact 
Jun-Hyeong Lee

yjhboky@gmail.com
