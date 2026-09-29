# Amazon ML Challenge 2026 — Entity Resolution

> An end-to-end machine learning pipeline for large-scale entity resolution, candidate generation, ranking, and final match prediction.

## Overview

This repository contains our solution for the **Amazon ML Challenge 2026**.

The challenge involved building a robust entity-matching system capable of identifying whether records from different sources referred to the same underlying entity. The main focus of our solution was to design a scalable retrieval-and-ranking pipeline that could handle a large candidate space while maintaining high matching recall and precision.

Our approach evolved through multiple experiments, progressively improving candidate generation, feature engineering, ranking, inference, and decision policies.

## Problem

Entity resolution is the task of determining whether two records represent the same real-world entity despite differences in:

* Names
* Addresses
* Contact information
* Formatting
* Missing values
* Typographical variations
* Data-source inconsistencies

A naive all-pairs comparison approach becomes computationally expensive as the dataset grows.

Therefore, the solution was designed as a multi-stage pipeline:

```text
Raw Data
   ↓
Data Cleaning & Normalization
   ↓
Candidate Generation
   ↓
Feature Engineering
   ↓
Candidate Ranking
   ↓
Matching Policy
   ↓
Final Predictions
```

## Solution Approach

### 1. Data Processing

The input data was cleaned and normalized before matching.

Key processing steps included:

* Text normalization
* Missing-value handling
* Field-level preprocessing
* Standardization of entity attributes
* Construction of matching features

### 2. Candidate Generation

Instead of comparing every possible pair, the pipeline first generated a smaller set of plausible candidates.

This stage was designed to maximize **candidate recall** while keeping the search space computationally manageable.

Multiple candidate-generation strategies were experimented with during development.

### 3. Feature Engineering

Candidate pairs were represented using multiple complementary signals, including:

* String similarity
* Token-level similarity
* Numerical similarity
* Structural features
* Field-level matching signals
* Cross-field consistency

These features were then provided to the ranking model.

### 4. Candidate Ranking

A learning-to-rank / classification-based approach was used to distinguish stronger matches from weaker candidates.

The ranking stage allowed the system to prioritize the most likely entity match instead of relying on a single deterministic similarity score.

### 5. Matching Policy

The final decision layer converted model scores and candidate rankings into the required submission format.

This stage was separately evaluated because a candidate can be retrieved successfully but still be rejected or ranked incorrectly during final inference.

## Experiment Evolution

The solution was developed iteratively through multiple versions.

```text
V1
 ↓
Initial matching pipeline
 ↓
V2
 ↓
Improved candidate generation
 ↓
V3
 ↓
Feature / ranking improvements
 ↓
V4
 ↓
Large-scale inference and optimization
 ↓
V5
 ↓
Final candidate + ranking + policy pipeline
```

Each version was evaluated independently before being incorporated into the final pipeline.

## Evaluation

A major focus during development was separating different sources of matching failure:

```text
Ground Truth Match
        │
        ├── Not generated as candidate
        │       └── Candidate Generation Error
        │
        └── Generated as candidate
                │
                ├── Ranked incorrectly
                │       └── Ranking Error
                │
                └── Correctly ranked
                        │
                        └── Final Matching Policy
```

This decomposition made it possible to identify whether performance limitations originated from retrieval, ranking, or the final decision policy.

## Final Pipeline

The final inference workflow was designed to process the complete evaluation data while maintaining reproducibility.

```text
Input Records
     ↓
Normalization
     ↓
Candidate Generation
     ↓
Feature Computation
     ↓
Model Scoring
     ↓
Candidate Ranking
     ↓
Matching Policy
     ↓
Submission Generation
```

## Repository Structure

```text
amazon-ml-challenge/
│
├── experiments/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   └── v5/
│
├── candidate_generation/
│
├── ranking/
│
├── features/
│
├── inference/
│
├── evaluation/
│
├── submission/
│
├── configs/
│
├── scripts/
│
└── README.md
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* LightGBM / ranking models
* PyTorch
* Git & GitHub

## Key Engineering Challenges

### Large Candidate Space

Generating candidates efficiently was critical because exhaustive pairwise comparison does not scale well.

### Recall vs. Computation

Candidate generation had to balance:

* High recall
* Candidate-set size
* Runtime
* Memory consumption

### Ranking Quality

Retrieving a true candidate is only the first step. The ranking model must subsequently place the correct candidate sufficiently high.

### Large-Scale Inference

The final pipeline required careful handling of:

* Memory
* Batch processing
* Checkpointing
* Intermediate artifacts
* Long-running inference jobs

## Key Learnings

This challenge provided practical experience in building an ML system beyond simply training a model.

The major lessons were:

1. **Candidate generation is as important as the ranking model.**
2. **Recall should be measured at every stage of the pipeline.**
3. **A strong model cannot recover a match that retrieval never produces.**
4. **Evaluation should separate retrieval, ranking, and policy errors.**
5. **Large-scale inference requires engineering considerations in addition to model quality.**
6. **Experiment tracking makes it easier to identify which changes actually improve the system.**

## Reproducibility

Each major experiment was maintained separately so that changes to candidate generation, ranking, and inference could be evaluated without losing previous results.

The final submission artifacts were generated through the corresponding final inference pipeline.

## Competition

**Amazon ML Challenge 2026**

This project represents our complete journey from initial entity-matching experiments to a production-oriented large-scale inference pipeline.

---

### Built with

**Machine Learning · Entity Resolution · Candidate Generation · Learning to Rank · Information Retrieval · Python**
