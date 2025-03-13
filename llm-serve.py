from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/pythia-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tells Accelerate to load only partial shards in GPU memory 
#   and the rest in CPU or disk offload as needed
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # partial GPU usage, rest on CPU
    offload_folder="./offload", # if you want disk-based offload
    torch_dtype="auto"
)

# Now you can run partial inference,
#   and HF will load shards dynamically into GPU as needed
