import os
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder2-15b"

# Fetch the current working directory and set the cache directory
cwd = os.getcwd()  # Get current working directory
cache_dir = os.path.join(cwd, "models")  # Ensure it's an absolute path

# Load the tokenizer and model from the custom cache directory
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)

print(f"Cache directory set to: {cache_dir}")
