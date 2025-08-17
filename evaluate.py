
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig 

base_model = "meta-llama/Llama-3.2-3B"
fine_tuned_model_path = "./results/checkpoint-848"  # Specific checkpoint

# Quantization config
compute_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Loading the base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}  
)

# Loading the fine-tuned LoRA adapters
fine_tuned_model = PeftModel.from_pretrained(model, fine_tuned_model_path)
fine_tuned_model.eval()  

tokenizer = AutoTokenizer.from_pretrained(base_model,token=secret_value_0, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

sample_input = {
    "Input": "let str = 'abc'; console.log(str.slice(2, 1));",
    "Suggested Fix": "let str = 'abc'; console.log(str.slice(1, 2));",
}

formatted_input = f"<s>[INST] Input: {sample_input['Input']}\nSuggested Fix: {sample_input['Suggested Fix']}\nProvide a review comment. [/INST]"


inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = fine_tuned_model.generate(
        **inputs,
        max_new_tokens=11,  
        do_sample=True,     
        top_p=0.9,          
        temperature=0.7     
    )


review_comment = tokenizer.decode(output[0], skip_special_tokens=True)

review_comment = review_comment.split("[/INST]")[-1].strip()

print("Generated Review Comment:", review_comment)