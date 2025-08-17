# JavaScript Code Review LLM Fine-Tuning  

This repository contains scripts and a dataset for **fine-tuning a Large Language Model (LLM)** to generate **automated code review comments** for JavaScript code snippets.  

The base model used is **Meta’s Llama-3.2-3B**, fine-tuned using **LoRA (Low-Rank Adaptation)** for efficiency. The fine-tuned model learns to provide reviewer-style comments based on:  

- Input JavaScript code  
- Error type (e.g., Syntax, Formatting, Best Practice, etc.)  
- Suggested fix  
- Human-like reviewer comments  

This project demonstrates how to:  

- 📑 Prepare and format a custom dataset for supervised fine-tuning (SFT).  
- ⚡ Fine-tune the model using **PEFT** and **BitsAndBytes** for quantization.  
- ✅ Evaluate the model by generating **review comments** on new inputs.  

It can be useful for **developers, code reviewers, or tools** integrating AI-assisted code reviews in JavaScript projects.  

---

## 📊 Dataset  

The dataset is provided in **`javascript_dataset_fully_unique.xlsx`** (Excel format, single sheet named `Code Review`).  
It consists of **997 unique entries** focusing on JavaScript review scenarios.  

### Dataset Structure  

| Column          | Description |
|-----------------|-------------|
| **Input**       | JavaScript code snippet with potential issues |
| **Error Type**  | Category (Syntax, Formatting, Best Practice, Logical, Reference Error, Type Error, Others) |
| **Suggested Fix** | Corrected version of the code |
| **Reviewer Comment** | Human-like review explanation |

### Dataset Statistics  

- **Total Rows**: 997  
- **Error Type Distribution**:  
  - Syntax: ~20%  
  - Formatting: ~40%  
  - Best Practice: ~15%  
  - Logical: ~10%  
  - Reference Error: ~5%  
  - Type Error: ~5%  
  - Others: ~5%  

**Example Entry:**  

| Input | Error Type | Suggested Fix | Reviewer Comment |
|-------|------------|---------------|------------------|
| `console.log('JavaScript is fun')` | Syntax | `console.log('JavaScript is fun');` | Missing semicolon. Add a semicolon at the end. |

---

## ⚙️ Requirements  

- Python **3.12+**  
- PyTorch (with CUDA for GPU acceleration)  
- Hugging Face libraries: `transformers`, `datasets`, `peft`, `trl`, `accelerate`  
- Quantization: **BitsAndBytes**  
- Additional: `pandas`, `scikit-learn`, `nltk`, `rouge_score`  

Install dependencies:  

```bash
pip install torch transformers datasets peft trl accelerate bitsandbytes pandas scikit-learn nltk rouge_score
```

### Hardware Recommendations  

- GPU with **16GB+ VRAM** (e.g., NVIDIA A100, RTX 3090)  
- Tested on **Kaggle** and **Google Colab** with GPU support  

---

## 🚀 Usage  

### 1. Setup  

Clone the repository:  

```bash
git clone https://github.com/nabeeljamshed23/PR_CodeReviewer.git
cd PR_CodeReviewer
```

Download the dataset (`javascript_dataset_fully_unique.xlsx`) and place it in the root directory.  

Obtain a **Hugging Face access token** for Llama models (set as environment variable or via Kaggle Secrets).  

---

### 2. Training the Model  

The **`train.py`** script handles:  

- Data loading & prompt formatting  
- Model quantization  
- LoRA configuration  
- Supervised fine-tuning with **SFTTrainer**  

**Key Configurations:**  

- **Base Model:** `meta-llama/Llama-3.2-3B`  
- **Quantization:** 4-bit (NF4) with float16 compute dtype  
- **LoRA Params:** rank=64, alpha=16, dropout=0.1  
- **Training Args:** 4 epochs, batch size=4, LR=1e-4, AdamW, FP16 mixed precision  
- **Dataset Split:** 85% train / 15% eval  
- **Prompt Format:**  

```
<s>[INST] Input: {input}
Suggested Fix: {fix}
Provide a review comment. [/INST] {comment}
```

Run training:  

```bash
python train.py
```

**Outputs:**  

- Checkpoints saved in `./results/` every 50 steps  
- TensorBoard logs for monitoring  
- Evaluation every 100 steps  

---

### 3. Evaluation & Inference  

The **`evaluate.py`** script loads a fine-tuned checkpoint and generates review comments.  

**Steps:**  

- Loads base model with quantization  
- Applies LoRA adapters from checkpoint  
- Formats input prompts like training  
- Generates reviewer comments with sampling  

Run evaluation:  

```bash
python evaluate.py
```

**Example Input:**  

```text
Input: let str = 'abc'; console.log(str.slice(2, 1));
Suggested Fix: let str = 'abc'; console.log(str.slice(1, 2));
```

**Example Output:**  

```
"The slice method parameters are incorrect as the start index is greater than the end index, which returns an empty string. Swap the indices to get the desired substring."
```

---

## 📈 Results  

- **Training Time:** ~1–2 hours on high-end GPU  
- **Eval Loss:** ~0.5–1.0 after 4 epochs  
- **Generated Comments:** Concise, context-aware, aligned with dataset patterns  

**Limitations:**  

- May hallucinate on **out-of-distribution inputs**  
- Larger datasets or further tuning recommended  

---

## 🤝 Contributing  

Contributions are welcome! You can:  

- Add more dataset entries  
- Improve evaluation metrics (BLEU/ROUGE integration)  
- Test on other base models  

Submit **issues** or **pull requests** on GitHub.  

