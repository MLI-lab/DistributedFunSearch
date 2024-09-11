from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

#huggingface_token = "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"  # for Llama 3.8
huggingface_token = "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" # for Llama Code
login(token=huggingface_token)

checkpoint = "meta-llama/Meta-Llama-3.1-8B-Instruct"
checkpoint2 = "meta-llama/CodeLlama-13b-Python-hf"
checkpoint3 = "bigcode/starcoder2-15b"


cache_dir = "/franziska/ChachingFace"

# Load the tokenizer and model from the custom cache directory
#tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
#model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(checkpoint2, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint2, cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(checkpoint3, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint3, cache_dir=cache_dir)
