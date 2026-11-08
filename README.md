# Consumer Complaint Classification

End-to-end NLP project that predicts the product category of a consumer complaint from its narrative. It compares three recurrent models built from scratch (SimpleRNN, LSTM, and GRU) with a fine-tuned DistilBERT transformer, then demonstrates confidence-based routing in a Gradio app.

## Highlights

- **Four-model comparison:** SimpleRNN, LSTM, GRU, and fine-tuned DistilBERT.
- **Fair model selection:** stratified train/validation/test split; the deployment model is selected by validation macro-F1, and the test set is reserved for final reporting.
- **Separate text pipelines:** cleaned and lemmatized text for the recurrent models, lightly cleaned natural text for DistilBERT.
- **Imbalance-aware training:** training is capped at 8,000 complaints per category, while validation and test retain the natural category distribution.
- **Routing analysis:** estimates automatic coverage and accuracy at several confidence thresholds, sending the remaining complaints to a person.
- **Interactive demo:** Gradio interface displays predicted category and per-category probabilities.

## Dataset

The notebook uses the [Consumer Complaints Dataset for NLP on Kaggle](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp). The CSV contains a complaint narrative (`narrative`) and product category (`product`). The dataset is not included in this repository; attach it as an input to the Kaggle notebook.

The saved run started with 162,421 rows. The five categories below were retained:

| Product category | Complaints | Share |
|---|---:|---:|
| credit_reporting | 91,172 | 56.14% |
| debt_collection | 23,148 | 14.25% |
| mortgages_and_loans | 18,990 | 11.69% |
| credit_card | 15,566 | 9.58% |
| retail_banking | 13,535 | 8.33% |

| Stage | Complaints |
|---|---:|
| Raw CSV rows | 162,421 |
| Rows with a narrative and label | 162,411 |
| After text cleanup and duplicate removal | 123,154 |
| Training used (8,000 per category) | 40,000 |
| Validation | 12,316 |
| Test | 12,316 |

The training set is capped after the stratified split. Validation and test keep the natural distribution, so test scores reflect the original category imbalance more closely than a balanced test set would.

## Results

The selected model is the one with the highest validation macro-F1. Values below are from the notebook's saved run on a Tesla T4 GPU.

| Model | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| **Fine-tuned DistilBERT (selected)** | **0.8400** | **0.8349** | **0.8428** | **0.8376** |
| LSTM | 0.8343 | 0.8253 | 0.8362 | 0.8281 |
| GRU | 0.8312 | 0.8213 | 0.8364 | 0.8282 |
| SimpleRNN | 0.8116 | 0.8020 | 0.8154 | 0.8067 |

### Test results by category — DistilBERT

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| credit_card | 0.753 | 0.859 | 0.802 | 1,498 |
| credit_reporting | 0.936 | 0.807 | 0.867 | 5,501 |
| debt_collection | 0.742 | 0.834 | 0.785 | 2,100 |
| mortgages_and_loans | 0.803 | 0.907 | 0.851 | 1,872 |
| retail_banking | 0.868 | 0.897 | 0.882 | 1,345 |
| **Macro average** | **0.820** | **0.861** | **0.838** | **12,316** |
| Weighted average | 0.853 | 0.843 | 0.844 | 12,316 |

### Confidence-based routing

The notebook reports how many test complaints meet each confidence threshold and how accurate those selected predictions are. This is a descriptive analysis; select and calibrate an operating threshold on validation data before using it for real routing.

| Confidence threshold | Routed automatically | Accuracy of routed cases | Sent to a person |
|---:|---:|---:|---:|
| 0.70 | 87.5% | 89.2% | 12.5% |
| 0.80 | 80.9% | 91.2% | 19.1% |
| 0.90 | 70.6% | 93.7% | 29.4% |
| 0.95 | 59.0% | 95.8% | 41.0% |

## Discussion

- DistilBERT achieved the best validation macro-F1 (0.8349) and test macro-F1 (0.8376). Its test macro-F1 is about 0.9 percentage points above the GRU, the strongest recurrent model by test macro-F1.
- The recurrent models are much smaller in this run: about 1.5–1.7 million parameters versus 67.0 million for DistilBERT. The notebook measured roughly 2.2 minutes to train the GRU and 9.5 minutes for DistilBERT; test inference throughput was about 25,392 versus 546 complaints/second, respectively. These timing figures depend on hardware and batch sizes.
- The largest class is credit_reporting (56.14% of the data), and the most frequent confusion for DistilBERT was credit_reporting predicted as debt_collection (510 complaints).
- The results use one random split and one seed. Small differences between models may be within run-to-run variation.
- The recurrent text pipeline removes stop words, which can discard negation cues. A useful next experiment is to preserve negations and compare against a TF-IDF + Logistic Regression or Linear SVM baseline.

## Run the notebook

The notebook is designed for Kaggle:

1. Open `consumer-complaint-classification.ipynb` in Kaggle.
2. Attach the Kaggle dataset linked above and enable a GPU accelerator.
3. Run the cells from top to bottom. The notebook searches Kaggle inputs for a CSV and detects the narrative and label columns.
4. If it chooses the wrong file or columns, set `CSV_PATH`, `TEXT_COLUMN`, and `LABEL_COLUMN` in the setup cell.

The notebook downloads NLTK corpora and the pretrained DistilBERT tokenizer/model while running, so internet access may be needed. The selected model and class labels are saved under `/kaggle/working`.

## Dependencies

See [`requirements.txt`](requirements.txt). PyTorch installation may need to match the CUDA/runtime environment when running outside Kaggle.

## Repository contents

- `consumer-complaint-classification.ipynb` — complete notebook with outputs from the reported run.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — excludes local data, checkpoints, and runtime files.
