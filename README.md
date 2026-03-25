# NLP_project_CIPV
Neuro-Symbolic CIPV Detection: A Graph-RAG Approach

![alt text](https://img.shields.io/badge/Domain-NLP-blue.svg)


![alt text](https://img.shields.io/badge/Architecture-Graph--RAG-green.svg)


![alt text](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)

This repository contains the implementation of a Context-Aware Graph-RAG pipeline designed for the automatic detection and forensic analysis of Cyber Intimate Partner Violence (CIPV).

Developed as part of the NLP Course 2024-2025 at the University of Bari Aldo Moro, this project bridges the gap between raw conversational data and structured expert knowledge using a neuro-symbolic approach.
📌 Project Overview

Traditional NLP toxicity classifiers often fail to detect subtle, non-profane patterns of abuse such as gaslighting, isolation, or economic control. Our system addresses these challenges by anchoring Large Language Models (LLMs) to a deterministic Knowledge Graph (KG) built on Neo4j.
Key Pillars:

    Clinical Grounding: Integrated with the SARA (Spousal Assault Risk Assessment) factors to evaluate recidivism and lethality risk.

    Legal Grounding: Anchored to the Italian "Codice Rosso" legislation for precise legal categorization (e.g., Art. 612-bis, 572).

    Context Awareness: Implements a Multi-scale Sliding Window (N=3, 6, 10) to capture the cumulative nature of coercive control.

⚙️ Architecture

The system follows a multi-stage Graph-RAG workflow:

    Preprocessing: Narrative flattening of JSON chat logs and lexical normalization.

    Hybrid Retrieval: Combines Dense Vector Search (using BAAI/bge-m3) with Sparse Entity Linking (via Llama-3.1-8B).

    Knowledge Injection: Expert nodes (Tactics, Laws, SARA factors) are retrieved from Neo4j and injected into the LLM prompt.

    Inference: Llama-3.1-8B-Instruct generates a structured forensic report with an explicit Confidence Score.

📊 Evaluation & Results

The pipeline was validated using an LLM-as-a-Judge protocol on a synthetic dataset of 300 chat windows.
Metric	Score (1-5)
Clinical Coherence	5.00 / 5.00
Legal Accuracy	5.00 / 5.00
Factual Grounding	1.09 / 5.00*
Overall Confidence	82.70%

*Note: The low Grounding score reflects the successful injection of external expert knowledge (Laws/SARA) not explicitly present in the raw source text.
🛠️ Tech Stack

    Language Model: Llama-3.1-8B-Instruct (via 4-bit Quantization)

    Database: Neo4j Aura Cloud (Knowledge Graph)

    Embeddings: BAAI/bge-m3 (1024-dim)

    Frameworks: Transformers, Accelerate, BitsAndBytes, Pandas

    Environment: Kaggle GPU (Tesla T4)

📁 Repository Structure

    /notebooks: Kaggle notebooks containing the full pipeline and ablation study.

    /ontology: Cypher scripts to build the Knowledge Graph in Neo4j.

    /report: The final project report (PDF) and LaTeX source.

🎓 Authors

    Federica Picca - [f.picca2@studenti.uniba.it]

    University of Bari Aldo Moro - Department of Computer Science
