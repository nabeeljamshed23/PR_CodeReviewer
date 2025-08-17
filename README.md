JavaScript Code Review LLM Fine-Tuning



This repository contains scripts and a dataset for fine-tuning a large language model (LLM) to generate automated code review comments for JavaScript code snippets. The base model used is Meta's Llama-3.2-3B, fine-tuned using LoRA (Low-Rank Adaptation) for efficiency. The fine-tuned model learns to provide helpful reviewer comments based on input code, error types, and suggested fixes.

The project demonstrates how to:





Prepare and format a custom dataset for supervised fine-tuning (SFT).



Fine-tune the model using PEFT (Parameter-Efficient Fine-Tuning) and BitsAndBytes for quantization.



Evaluate the model by generating review comments on sample inputs.

This can be useful for developers, code reviewers, or tools integrating AI-assisted code reviews in JavaScript projects.

Dataset

The dataset is provided in javascript_dataset_fully_unique.xlsx (Excel format with a single sheet named "Code Review"). It consists of ~997 unique entries focused on JavaScript code review scenarios. Each row includes:





Input: The original JavaScript code snippet with potential issues (e.g., syntax errors, formatting problems).



Error Type: Category of the issue (e.g., Syntax, Formatting, Best Practice, Logical, Reference Error, Type Error).



Suggested Fix: A corrected version of the code.



Reviewer Comment: A human-like review comment explaining the issue and why the fix is recommended.

Dataset Statistics





Total Rows: 997 (excluding header).



Error Type Distribution (approximate):





Syntax: ~20%



Formatting: ~40%



Best Practice: ~15%



Logical: ~10%



Reference Error: ~5%



Type Error: ~5%



Others: ~5%



Example Entry:







Input



Error Type



Suggested Fix



Reviewer Comment





console.log('JavaScript is fun')



Syntax



console.log('JavaScript is fun');



Missing semicolon. Add a semicolon at the end.

The dataset is fully unique, with no duplicates, and covers common JavaScript pitfalls like missing semicolons, improper spacing, type coercion, and logical errors. It was designed for training models to mimic code reviewers.

Requirements





Python 3.12+



PyTorch (with CUDA support for GPU acceleration)



Hugging Face Transformers, Datasets, PEFT, TRL, and Accelerate libraries



Access to Hugging Face Hub (for model download; requires a token for gated models like Llama-3.2-3B)



Kaggle Secrets (optional, for secure token management in environments like Kaggle)



Additional libraries: pandas, scikit-learn, nltk, rouge_score

Install dependencies via:

pip install torch transformers datasets peft trl accelerate bitsandbytes pandas scikit-learn nltk rouge_score

Hardware Recommendations:





GPU with at least 16GB VRAM (e.g., NVIDIA A100 or RTX 3090) for efficient training with 4-bit quantization.



Training was tested on environments like Kaggle or Colab with GPU support.

Usage

1. Setup





Clone the repository:

git clone https://github.com/your-username/javascript-code-review-llm.git
cd javascript-code-review-llm



Download the dataset (javascript_dataset_fully_unique.xlsx) and place it in the root directory (or update the file path in train.py).



Obtain a Hugging Face access token for Llama models (set as an environment variable or use Kaggle Secrets as shown in the scripts).

2. Training the Model

The train.py script handles data loading, prompt formatting, model quantization, LoRA configuration, and supervised fine-tuning using the SFTTrainer from TRL.

Key Configurations:





Base Model: meta-llama/Llama-3.2-3B



Quantization: 4-bit (NF4) with float16 compute dtype for memory efficiency.



LoRA: Rank=64, Alpha=16, Dropout=0.1.



Training Args: 4 epochs, batch size=4, learning rate=1e-4, AdamW optimizer, FP16 mixed precision.



Dataset Split: 85% train, 15% eval.



Prompt Format: <s>[INST] Input: {input}\nSuggested Fix: {fix}\nProvide a review comment. [/INST] {comment}

Run the training:

python train.py





Outputs: Checkpoints saved in ./results every 50 steps. TensorBoard logs for monitoring.



Training Time: ~1-2 hours on a high-end GPU (depending on dataset size and hardware).



Evaluation: Runs every 100 steps during training.

Notes:





Adjust num_train_epochs, per_device_train_batch_size, or learning_rate based on your hardware.



If using Kaggle, ensure the secret llamaAPI is set with your Hugging Face token.

3. Evaluation and Inference

The evaluate.py script loads a fine-tuned checkpoint and generates review comments for new inputs.

Key Steps:





Loads the base model with quantization.



Applies LoRA adapters from a checkpoint (e.g., ./results/checkpoint-848).



Formats input prompts similarly to training.



Generates output using sampling (top_p=0.9, temperature=0.7, max_new_tokens=11).

Run evaluation:

python evaluate.py





Sample Input (hardcoded in script):

Input: let str = 'abc'; console.log(str.slice(2, 1));
Suggested Fix: let str = 'abc'; console.log(str.slice(1, 2));



Expected Output: A generated review comment, e.g., "Incorrect slice indices; start should be less than end."

Customization:





Update sample_input in the script for new test cases.



Metrics: The script includes imports for BLEU and ROUGE scores (from nltk and rouge_score), but they are not used by default. Add evaluation loops to compare generated vs. ground-truth comments.

Example Output

Generated Review Comment: "The slice method parameters are incorrect as the start index is greater than the end index, which returns an empty string. Swap the indices to get the desired substring."

Results





After 4 epochs, the model generates coherent, context-aware review comments.



Eval Loss: Typically decreases to ~0.5-1.0 (monitor via TensorBoard).



Qualitative: Comments align well with dataset patterns, explaining errors and fixes concisely.



Limitations: Model may hallucinate on out-of-distribution inputs; further tuning or larger datasets recommended.

Contributing

Contributions are welcome! Feel free to:





Add more dataset entries.



Improve evaluation metrics (e.g., integrate BLEU/ROUGE fully).



Test on other base models.

Submit issues or pull requests on GitHub.
