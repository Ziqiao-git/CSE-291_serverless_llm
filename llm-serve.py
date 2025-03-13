# llm-serve.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "EleutherAI/pythia-1b"
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model with partial offload...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",       # partial GPU usage, rest on CPU
        offload_folder="./offload",
        torch_dtype="auto"
    )
    print("Model loaded. Doing a quick generation test...")

    prompt = "Hello from partial load!"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("Generated text:", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
