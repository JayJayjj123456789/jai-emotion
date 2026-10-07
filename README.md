---
language:
- th
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: airesearch/wangchanberta-base-att-spm-uncased
datasets:
- wisesight_sentiment
- wongnai_reviews
tags:
- sentiment-analysis
- emotion-classification
- thai
- wangchanberta
metrics:
- accuracy
widget:
- text: "โอโห้ ช่องนี้เปิดโลกเรามากเลยค่ะ ดูจีเนียสมากๆๆ"
  example_title: "Positive"
- text: "เริ่มจากชายเน็ตคนหนึ่งเปิดประเด็นว่าไปพบเจ้าจุดดำลึกลับนี้กลางมหาสมุทรใน Google Maps"
  example_title: "Neutral"
- text: "ผมเป็นคนที่ไม่มีความสุขเลยจริงๆ"
  example_title: "Negative"
---

# Jai Emotion: Thai 3-class Sentiment Classification

**Jai Emotion** is a Thai 3-class sentiment/emotion classifier built on
[WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased),
fine-tuned on social-media posts and product reviews. It predicts one of
`positive`, `neutral`, or `negative`.

Developed for the **JaiKraJok** project (ใจกระจก, "heart mirror") as a
self-hosted, dependency-free alternative to closed sentiment APIs.

## Model Description

- **Model type**: `CamembertForSequenceClassification` (WangchanBERTa base, 105M params)
- **Language**: Thai (th)
- **Labels**: `positive` (0), `neutral` (1), `negative` (2)
- **Base model**: `airesearch/wangchanberta-base-att-spm-uncased`
- **Training**: full fine-tune (100% of parameters), 5 epochs, best-checkpoint
- **License**: Apache 2.0
- **Requirement**: `transformers` 4.30+

## Performance

Evaluated on a held-out **Wisesight** test split (balanced, n=798) and an
in-domain Thai student-message set (n=100).

| Model | Params | thai_emotion (in-domain) | wisesight (held-out) |
|---|---|---|---|
| **Jai Emotion** | 105M | **100.0%** | **73.2%** |
| `poom-sci/WangchanBERTa-finetuned-sentiment` | 105M | 80.0% | 68.4% |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 278M | — | 56.8% |
| S-Sense (SSense 2.0, NECTEC) | — | 82.0% | 50.0% |

**Note on reported numbers.** Many Thai sentiment models report 90%+ accuracy;
those results are **binary** (positive/negative only, `neutral` excluded). This
model is **3-class**, which is substantially harder — neutral is the most
ambiguous and hardest class. The published 3-class ceiling on Wisesight is
≈ 73–76% accuracy (Macro-F1 ≈ 0.63); this model reaches it.

### Ablation

| Configuration | wisesight (held-out) |
|---|---|
| Partial fine-tune (frozen embeddings + 8/12 layers) | 63.8% |
| **Full fine-tune (all layers)** | **73.1%** |
| Full fine-tune + focal loss + multi-domain data | **73.2%** |

The dominant factor was **training scope**, not data volume: freezing the lower
layers capped accuracy ~9 points below a full fine-tune.

## Intended Uses & Limitations

**Intended use.** Classifying short Thai text (social posts, chat messages,
reviews) into positive / neutral / negative for downstream product features.

**Out-of-scope.** Long documents, formal/legal text, non-Thai text, and
fine-grained emotion taxonomies (this model collapses sadness, anger, and fear
into `negative`).

**Limitations.**
- Trained mainly on social-media and review text; domain shift degrades accuracy.
- `neutral` is inherently ambiguous — expect the majority of errors here
  (neutral↔positive confusions dominate).
- No class for questions (`?` posts) — these are mapped to `neutral`.
- Not evaluated on political, medical, or safety-critical content.

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "JayJayjj123456789/jai-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

text = "โอโห้ ช่องนี้เปิดโลกเรามากเลยค่ะ"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=160)
with torch.no_grad():
    logits = model(**inputs).logits
pred = logits.argmax(-1).item()
print(model.config.id2label[pred])  # -> positive
```

## Training Data

| Source | Size | Domain | Label mapping |
|---|---|---|---|
| `wisesight_sentiment` (train) | 23,545 | social media | pos/neu/neg |
| `Wongnai/wongnai_reviews` | 37,805 | restaurant reviews | 1–2→neg, 3→neu, 4–5→pos |
| thai_emotion (student messages) | ×20 oversample | in-domain | joy→pos, sad/anger/fear→neg |
| **Total** | **57,216** | | |

### Preprocessing

- Whitespace normalization; newlines collapsed to spaces.
- Reviews filtered to 10–1500 characters.
- No synthetic data in the final model (synthetic augmentation was tested and
  found to dilute the signal — see Ablation).
- Tokenized with the WangchanBERTa SentencePiece tokenizer, `max_length=160`.

### Training Procedure

| Hyperparameter | Value |
|---|---|
| Epochs | 5 (best-checkpoint by validation accuracy) |
| Batch size | 16 |
| Learning rate | 2e-5 (linear warmup 6%) |
| Optimizer | AdamW, weight decay 0.01 |
| Loss | Focal Loss (γ=2) + class weights α=[1.59, 0.56, 0.85] |
| Max sequence length | 160 |
| Hardware | Apple M5 (MPS), ~205 min |

## Citation

```bibtex
@misc{jaiemotion2025,
      title={Jai Emotion: A Thai 3-class Sentiment Classifier based on WangchanBERTa},
      author={JaiKraJok Team},
      year={2025},
      howpublished={\url{https://github.com/JayJayjj123456789/jai-emotion}}
}
```

Built on WangchanBERTa — please also cite:

```bibtex
@misc{lowphansirikul2021wangchanberta,
      title={WangchanBERTa: Pretraining transformer-based Thai Language Models},
      author={Lalita Lowphansirikul and Charin Polpanumas and Nawat Jantrakulchai and Sarana Nutanong},
      year={2021},
      eprint={2101.09635},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Files

```
config.json                    model configuration
model.safetensors              fine-tuned weights (402 MB)
tokenizer.json                 fast tokenizer
tokenizer_config.json          tokenizer configuration
sentencepiece.bpe.model        SentencePiece model
special_tokens_map.json        special tokens
logs/                          training + evaluation logs
```
