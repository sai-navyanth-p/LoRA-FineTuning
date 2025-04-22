# 🔍 LoRA Fine-Tuning on AG News Dataset using RoBERTa

---

## 👥 Team Members

- **Sai Navyanth Penumaka**: sp8138  
- **Karthik Sunkari**: ks7929  
- **Geethika Rao Gouravelli**: gg2879  

---

## 📄 Description

This project implements Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** on the AG_NEWS text classification dataset. Instead of full fine-tuning, LoRA updates only a small set of additional parameters, drastically reducing memory usage and training cost. The model is based on `roberta-base` and optimized for sequence classification tasks. The goal is to achieve high performance while fine-tuning a minimal number of parameters.

---

## 🔍 Overview

We applied **Low-Rank Adaptation (LoRA)** to fine-tune a frozen `roberta-base` model on the AG_NEWS dataset for news topic classification. The classification task involves predicting one of four news categories. The implementation supports early stopping, configurable settings, and evaluation using HuggingFace’s `Trainer` API. The configuration is fully modularized via a centralized `Settings` class.

LoRA modifies only specific modules (`query`, `value`) in attention layers using lightweight adapters. The rest of the model remains frozen, leading to reduced compute costs and faster training, with a strong generalization ability.

---

## 🧠 Summary (Steps / Model Architecture)

### ✅ Steps Followed:

1. Loaded AG_NEWS dataset from HuggingFace
2. Preprocessed and tokenized text using `RobertaTokenizer` (`max_length=64`, `padding=True`, `truncation=True`)
3. Initialized `roberta-base` as the base model
4. Applied LoRA with a small rank (r=3), modifying only attention module components
5. Trained the model for 1 epoch using HuggingFace `Trainer`
6. Early stopping enabled with patience set to 5
7. Evaluated performance on validation and test datasets
8. Saved best-performing model

### 🧱 Model Architecture:

- **Base Model**: `roberta-base`
- **Classification Head**: Default head from HuggingFace `AutoModelForSequenceClassification`
- **PEFT**: LoRA
  - Target Modules: `query`, `value`
  - Rank (`r`): 3
  - Alpha: 6
  - Dropout: 0.05
- **Loss Function**: Cross Entropy
- **Optimizer**: AdamW
- **Freezing Strategy**: Full model frozen except LoRA-injected modules

---

## 📊 Output

| Dataset     | Accuracy |
|-------------|----------|
| Validation  | 88.1%    |
| Test (Eval) | 85.625%  |

📌 **Trainable Parameters**: 667,396  
🎯 Only a fraction (~0.5%) of `roberta-base` parameters were fine-tuned

---

## ⚙️ Parameters

| Parameter                 | Value                        |
|---------------------------|------------------------------|
| Model Name                | `roberta-base`               |
| Max Sequence Length       | 64                           |
| Train Batch Size          | 32                           |
| Eval Batch Size           | 64                           |
| Epochs                    | 1                            |
| Learning Rate             | 5e-6                         |
| Weight Decay              | Disabled                     |
| Early Stopping            | Enabled (Patience = 5)       |
| LoRA Rank (`r`)           | 3                            |
| LoRA Alpha                | 6                            |
| LoRA Dropout              | 0.05                         |
| Target Modules            | `query`, `value`             |
| Task Type                 | Sequence Classification      |
| Total Trainable Params    | 667,396                      |
| Save Directory            | `my_best_model`              |
| Freeze Base Model         | True                         |
| Evaluation Strategy       | Every 500 steps              |
| Save Steps                | 4000                         |
| Logging Steps             | 100                          |
| Scheduler                 | Linear                       |
| Warmup Ratio              | 0.1                          |
| Reporting Tool            | Weights & Biases (wandb)     |

---

> 🚀 This project shows how powerful LoRA is when it comes to fine-tuning large transformer models like RoBERTa in a memory- and compute-efficient way. Even with less than 700K trainable parameters, the model achieves over 88% validation accuracy on AG_NEWS.
