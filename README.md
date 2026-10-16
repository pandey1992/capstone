# capstone
Mental Health concern classifier

# Capstone 4: Mental Health Concern Classifier

This repository contains the design specification and notebook for the Capstone 4 NLP classification project.

## Files

- `code.ipynb` - end-to-end workflow for TF-IDF + Logistic Regression, Word2Vec + Logistic Regression, Azure LLM classification, and fine-tuned DistilBERT.
- `Capstone_4_Design_Spec.md` - requirements map, implementation plan, rubric checklist, and submission checks.

## Running the notebook

The notebook targets the course Databricks Serverless GPU environment. It expects the course-provided data, offline DistilBERT assets, and Databricks secrets for Azure authentication. Keep data files and credentials out of GitHub. Locally named inputs with `.csv.txt` suffixes are also detected by the notebook.

Run all cells in order. The notebook reports validation metrics, chooses the model by macro-F1 (accuracy as the tie-breaker), then generates and validates `submission.csv`. The current notebook has not been executed end-to-end in this repository workspace.

Predictions are for the educational assignment and do not provide a clinical diagnosis.

