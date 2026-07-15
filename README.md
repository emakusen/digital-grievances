# Grievance Archetypes

Computational framework for discovering **grievance archetypes** in online discourse using **Gaussian Mixture Models (latent profile analysis)**, **LIWC-22 psycholinguistic features**, and **Moral Foundations Theory**.

## Overview

This repository accompanies the paper:

> **Grievance Archetypes: Identifying Psycholinguistic Patterns of Perceived Injustice in Online Discourse**

The pipeline identifies recurring psychological patterns in grievance expressions across different contexts by clustering comments based on psycholinguistic and moral features.

## Features

- Data preprocessing and filtering
- Feature extraction using LIWC-22 and Moral Foundations Theory
- Latent profile analysis with Gaussian Mixture Models
- Automatic model selection using BIC, entropy, minimum cluster size, and bootstrap stability (ARI)
- Archetype profiling
- Validation against grievance context and subcategories
- Export of representative comments
- Publication-ready figures and summary tables

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- joblib

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib
```

## Input

The input is a CSV file containing:

- comment text
- grievance context
- grievance subcategories
- moral foundation scores
- LIWC-22 features

Column names can be configured at the top of `grievance_archetypes.py`.

## Usage

Run the full pipeline:

```bash
python grievance_archetypes.py --input data.csv --stability
```

Specify the number of archetypes manually:

```bash
python grievance_archetypes.py --input data.csv --k 6
```

## Outputs

The pipeline generates:

- `model_selection.csv`
- `bic_curve.png`
- `archetype_profiles.csv`
- `profile_heatmap.png`
- `archetype_assignments.csv`
- `confidence_summary.csv`
- `validation_crosstab_*.csv`
- `top_comments_per_archetype.csv`

## Dataset
A sample dataset is provided for testing purposes. The dataset counts 5000 samples randomly sampled from our 1.4 million grievance dataset extracted from YouTube and paraphrased via Qwen2.5:7b-instruct using the following prompt:

```bash
Rewrite this YouTube comment as if it was written by another person.

Rules:
- Preserve the exact meaning.
- Preserve the grievance.
- Preserve sentiment and emotional intensity.
- Preserve insults, profanity, sarcasm, and slang if present.
- Do not add new information.
- Do not remove information.
- If the original comment is not English, rewrite it naturally in English.
- Do not explain.
- Do not use quotation marks.
- Return only the rewritten comment.
```

## Citation

If you use this repository, please cite:

```bibtex
@article{kahr2026grievance,
  title={Grievance Archetypes: Identifying Psycholinguistic Patterns of Perceived Injustice in Online Discourse},
  author={Kahr, Ema and Plavyuk, Anna},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
