# Capstone 4 Design Specification

## Objective

Build a reproducible binary text classifier for short social-media-style posts. Predict `1` for a potential anxiety or stress concern and `0` for no concern detected. The capstone deliverables are an executed `code.ipynb` notebook and `submission.csv`, containing one integer prediction per test row in the original test order.

This design is mapped to the learner guide's four 3-point rubric categories (12 points maximum): code quality, classical NLP, advanced NLP, and model performance. Target the **Excellent** level in every category. The performance rubric sets a target of macro-F1 >= 0.77 and accuracy >= 0.77 on the hidden test set; this is a target, not a guarantee.

## Inventory and observed state

| File | Purpose / observation |
|---|---|
| `Capstone 4 Learner Instructions.pdf` | Governing assignment, workflow, deliverables, rubric, and checklist. |
| `Capstone_Instructions.md.txt` | Runtime, offline model/data paths, LLM configuration, checkpoint, and submission notes. |
| `cap4_train_val.csv.txt` | 1,500 rows; columns `text`, `label`; no nulls or blank texts; labels balanced (750 each); text length median 80 words, 99th percentile ~193; 4 repeated text values and 3 duplicate full rows. |
| `cap4_test.csv.txt` | 1,000 rows; one `text` column; no nulls or blank texts; text length median 81 words, 99th percentile ~178; 4 repeated text values. Preserve its row order. |
| `UAIS_vars.env.txt` | Azure OpenAI endpoint/deployment/API version/project configuration for the LLM workflow. Keep credentials in Databricks secrets; do not place secrets in the notebook or submission. |

The local folder does **not** currently contain `code.ipynb`, `Submit_Capstone_Project`, `.setup/learner_setup`, or the referenced `/Workspace/advanced_data/...` assets. The project notes say the hosted Databricks environment supplies these. Confirm the notebook and expected data/model paths are available in that environment before implementation. The brief references `UAIS_vars.env`, while the local copy is named `UAIS_vars.env.txt`; resolve that path/name in the notebook environment without committing secrets.

The two dataset filenames end in `.csv.txt` in this folder, although the assignment refers to `.csv`. Read the actual available paths or make an explicit notebook-side copy/rename in the permitted workspace. Do not modify source data silently.

## Binding requirements

1. Use only `text` as the feature and `label` as the target.
2. Create one stratified 80/20 split from the 1,500-row labeled file, with `random_state=42` (expected 1,200 train / 300 validation). Reuse the same indices across every approach.
3. Set reproducible seeds to 42 for Python, NumPy, scikit-learn, and PyTorch/Transformers where applicable. Record relevant library/model configuration.
4. Fit learned preprocessing and models on the training split only. Validation is for model evaluation and limited tuning; test remains untouched until model selection is finalized.
5. Implement and compare all four mandated approaches:
   - TF-IDF + Logistic Regression.
   - Word2Vec document-average embeddings + Logistic Regression.
   - AzureChatOpenAI (`gpt-4.1-mini`) prompted as a classifier.
   - Fine-tuned local DistilBERT sequence classifier.
6. For each approach, report validation macro-F1, accuracy, confusion matrix, and classification report, followed by a short interpretation.
7. Select one model using validation macro-F1 first, accuracy as tie-breaker. Document the choice and avoid using test outcomes to choose or tune.
8. Generate integer test predictions and save `submission.csv` with exactly one column `y_pred`, no index or other columns, and exactly 1,000 rows in original test order.
9. Deliver `code.ipynb` with clear Markdown sections and all cells executed with outputs retained. Final submission should contain only the notebook and required prediction CSV, per the guide.

## Notebook design

### 1. Overview and setup

- State the classification objective, label interpretation, metric priority, and leakage safeguards.
- Import dependencies, display versions, and configure deterministic seeds.
- Run the provided setup only if it is present/required in the hosted workspace (`%run ./.setup/learner_setup`).
- Load NLTK resources from `/Workspace/advanced_data/nltk`, PyTorch assets from `/Workspace/advanced_data/torch`, and DistilBERT from `/Workspace/advanced_data/distilbert-model`; do not attempt internet model downloads in the offline serverless environment.
- Load Azure settings from the provided environment configuration and obtain client credentials through Databricks secrets as specified. Never print tokens/secrets.
- Use `/tmp` for model checkpoints to avoid serverless write failures; clean up only temporary artifacts after they are no longer needed.

### 2. Data loading and split

- Read the labeled and test files; assert expected columns, row counts, non-null text, binary labels, and non-empty strings.
- Preserve test row order and retain a stable positional index for final output checks.
- Print label balance and basic text-length summaries. Inspect duplicate counts and whether exact duplicate texts cross the train/validation boundary; document this as a data-quality observation. Do not use test labels (none exist) or test text to fit any vocabulary/model.
- Create one `train_test_split` with `test_size=0.20`, `stratify=y`, `random_state=42`. Persist train/validation positions and reuse throughout.

### 3. Preprocessing policy

Keep separate model-specific text views:

- **Classical:** lowercase, normalize whitespace, remove URLs and optionally email/@handle artifacts. Tokenize consistently. Compare a justified light-cleaning variant with any more aggressive removal only on training-fit/validation evaluation. Do not remove numbers, punctuation, or stop words by default when they can signal context; if testing removal, report the reason and validation effect. Avoid heavy stemming/lemmatization absent demonstrated benefit.
- **LLM:** retain natural wording; only remove obvious disruptive URL artifacts or normalize whitespace. No stop-word deletion or aggressive normalization.
- **DistilBERT:** keep original wording with only essential cleanup; let the pretrained tokenizer handle tokenization.
- Keep every transform deterministic and stateless unless learned strictly from training data.

### 4. Classical NLP experiments

#### TF-IDF + Logistic Regression

- Use a scikit-learn `Pipeline` where practical so vectorization and classifier fit are auditable as one train-only operation.
- Establish a baseline with word unigrams/bigrams (`ngram_range=(1,2)`), a modest document-frequency threshold, and Logistic Regression (`max_iter` sufficient for convergence).
- Run a small, recorded validation-only comparison: n-grams `(1,1)`, `(1,2)`, optionally `(1,3)`; reasonable `min_df`/`max_df`; Logistic Regression `C` over a small range such as 0.1, 1, 10. Do not tune an excessive grid.
- Select the variant by macro-F1, then accuracy. Fit only on training; use the chosen pipeline unchanged for validation and later test.

#### Word2Vec average embeddings + Logistic Regression

- Tokenize/clean using the classical text view.
- Train gensim Word2Vec on training-split token sequences only. Record `vector_size`, `window`, `min_count`, `epochs`, `sg=1`, and seed/workers settings.
- Compare a small set of meaningful settings (for example, vector sizes 100/200 and 10/20 epochs, within compute limits). Keep `min_count` low enough for this small corpus and report out-of-vocabulary behavior.
- Create each document vector as the mean of available token vectors; use a zero vector for an empty/OOV-only document and count such cases.
- Train Logistic Regression only on training document vectors. Select using the same validation metrics.

### 5. Advanced NLP experiments

#### Prompted LLM classifier

- Use LangChain `AzureChatOpenAI` with the provided `gpt-4.1-mini` deployment/configuration and Databricks secret-backed token acquisition.
- Prompt must define the binary task precisely, distinguish “potential concern” from “no concern detected,” prohibit diagnosis or extra commentary, and demand a single character `0` or `1` only.
- Optional few-shot examples must come only from the training split, be balanced, and have no more than five examples per class. Explain why examples were selected. A zero-shot baseline is useful as a controlled comparator.
- Evaluate the full validation split. Parse outputs strictly, record malformed outputs, and apply one predeclared retry/repair rule; never silently coerce arbitrary prose into a class.
- Add bounded retry/backoff and checkpoint completed validation responses so transient errors do not force a full rerun. Keep prompts/outputs free of unnecessary personal details in logs. Record model/deployment, prompt version, and temperature.
- Include this workflow in the comparison even if it is slower or underperforms. Do not call the LLM on test until the final model selection is complete.

#### Fine-tuned DistilBERT

- Load the pre-downloaded tokenizer and `AutoModelForSequenceClassification` from `/Workspace/advanced_data/distilbert-model`, with `num_labels=2` and local-only loading.
- Use numeric labels directly. Tokenize with truncation and a justified maximum length (start around 256; validate the observed length coverage and use no more than 512). Use dynamic padding with a data collator.
- Fine-tune only on the training split. Use validation only for evaluation/checkpoint selection. Start with a modest, documented configuration, e.g. learning rate in `3e-5`–`1e-4`, batch size 8–32 subject to GPU memory, and 2–4 epochs. Use `load_best_model_at_end`/best macro-F1 checkpoint if supported and configured consistently with evaluation/save strategy.
- Save checkpoints under `/tmp`. Track validation metrics each epoch; avoid test evaluation during tuning.
- Since the runtime is offline, verify all tokenizer/model assets are local before training and fail clearly rather than attempting external downloads.

### 6. Evaluation and model selection

- Centralize evaluation in one helper to ensure consistent label order `[0,1]` and metric calculations.
- For each final candidate, report macro-F1 (`average='macro'`), accuracy, a labeled confusion matrix, and `classification_report` with zero-division handling.
- Summarize candidates in a comparison table with key settings, validation macro-F1, accuracy, and notes about errors/tradeoffs. Explain false positives and false negatives in this triage-style task without claiming clinical diagnosis.
- Use a small number of validation-driven iterations and keep an experiment log (configuration, change, result). The validation set is not for training or fitting preprocessing.
- Select the highest macro-F1 candidate; use accuracy only to break a tie or clarify a close result. State the final rationale.

### 7. Final prediction and submission checks

- After selection is frozen, refit the selected classical model on the full labeled 1,500 records (including prior validation) before test inference, if the chosen approach is classical. For DistilBERT, either retain the best train-only checkpoint as evaluated or perform a clearly documented final refit with the selected configuration on all labeled rows; do not use test labels. For LLM prompting, apply the frozen prompt to test rows. The notebook must distinguish validation evaluation from final refit.
- Apply the selected preprocessing and fitted components unchanged to test texts; preserve positional order.
- Assert prediction count is 1,000, values are integer 0/1, dataframe columns equal `['y_pred']`, and output order matches the loaded test dataframe.
- Save with `index=False` as `submission.csv`. Re-read it and assert one column, correct row count, no nulls, valid labels, and no index column.
- Do not touch or delete any generated CSV used by the submission automation; follow the notebook's submission workflow and terminate GPU compute when finished.

## Excellence checklist (12/12 target)

### Code quality - 3 points

- [ ] Modular functions for preprocessing, fitting/prediction, metrics, and output validation.
- [ ] Clear section headings and short rationale for design choices.
- [ ] Consistent names, concise comments, useful assertions, and understandable error messages.
- [ ] Seed 42 applied consistently; package/model settings recorded.
- [ ] Notebook executed end-to-end with visible outputs; no stale failed cells.

### Classical NLP - 3 points

- [ ] Shared stratified 80/20 split, random state 42.
- [ ] TF-IDF and Logistic Regression implemented; n-gram/DF/classifier settings reasonably compared.
- [ ] Word2Vec trained only on training tokens; average vectors correctly built for all splits.
- [ ] Validation metrics, confusion matrix, classification report, and meaningful comparison are present for both.
- [ ] Learned vectorizers/embeddings are never fitted on validation or test data.

### Advanced NLP - 3 points

- [ ] AzureChatOpenAI gpt-4.1-mini prompt classifier implemented and evaluated.
- [ ] Prompt clearly defines labels and strict one-character output; any few-shot examples are balanced and training-only (max five per class).
- [ ] DistilBERT loaded locally, trained only on training split, truncated appropriately, and dynamically padded.
- [ ] Both advanced workflows include all required validation metrics and a short interpretation.

### Model performance - 3 points

- [ ] Best model selected from validation results by macro-F1, accuracy secondary.
- [ ] Selection rationale is clear and tied to observed metrics and errors.
- [ ] Hidden-test performance target: macro-F1 >= 0.77 and accuracy >= 0.77; cannot be known locally without labels.
- [ ] Final CSV format/order/labels verified. Do not claim a test score without available test labels/evaluation results.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Notebook/runtime assets absent from local folder | Confirm execution in the provided Databricks notebook/environment before implementing local assumptions. |
| Offline serverless runtime attempts internet downloads | Load NLTK, PyTorch, and DistilBERT only from the documented local paths; use local-only model loading. |
| LLM latency, rate limits, or transient failures | Bound retries, checkpoint responses, track malformed outputs, and complete validation before final test inference. |
| Overfitting a 300-row validation split | Keep tuning small, use macro-F1 first, report uncertainty qualitatively, and avoid repeated prompt/model tweaking against individual cases. |
| Short/rare-word corpus weakens Word2Vec | Check OOV/zero-vector counts, keep `min_count` appropriate, and compare sensible dimensions/epochs. |
| Duplicate posts inflate validation scores | Report exact duplicate overlap between train and validation; interpret metrics with that limitation in mind. |
| Hidden-test score cannot be observed during development | Use validation as the sole selection basis; do not infer hidden-test performance from local file contents. |
| Submission CSV shape/order errors | Assert schema, count, integer labels, and row order before writing and after re-reading. |

## Definition of done

The notebook implements and documents all four approaches, evaluates each on the same validation split with all required metrics, selects one model by the prescribed rule, produces and validates a 1,000-row `submission.csv`, and is fully executed with outputs present. The final submission follows the course workflow and contains the notebook and required prediction file.

