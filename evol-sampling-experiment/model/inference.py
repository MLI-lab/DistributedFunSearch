# model/inference.py
"""
Model loading and inference.

Use vLLM for fast batched generation if available, else HuggingFace.
Track token usage for compute fairness analysis.
"""

from typing import Optional
import sys


class ModelWrapper:
    """Unified interface for vLLM and HuggingFace models."""

    def __init__(
        self,
        model_name: str,
        use_vllm: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.temperature = temperature
        self.max_tokens = max_tokens

        if use_vllm:
            self._init_vllm()
        else:
            self._init_huggingface()

    def _init_vllm(self):
        """Initialize vLLM model."""
        from vllm import LLM, SamplingParams

        print(f"Loading model with vLLM: {self.model_name}")
        self.model = LLM(model=self.model_name, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        print("Model loaded successfully")

    def _init_huggingface(self):
        """Initialize HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model with HuggingFace: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded successfully")

    def generate_solution(self, problem: str) -> dict:
        """
        Generate a solution for a math problem.

        Returns:
            {
                "response": str,
                "input_tokens": int,
                "output_tokens": int
            }
        """
        prompt = self._make_solution_prompt(problem)
        return self._generate(prompt)

    def generate_refinement(self, problem: str, previous_solution: str) -> dict:
        """
        Generate a refined solution given a previous attempt.

        Returns:
            {
                "response": str,
                "input_tokens": int,
                "output_tokens": int
            }
        """
        prompt = self._make_refinement_prompt(problem, previous_solution)
        return self._generate(prompt)

    def _make_solution_prompt(self, problem: str) -> str:
        """Create prompt for initial solution generation."""
        return f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

    def _make_refinement_prompt(self, problem: str, previous: str) -> str:
        """Create prompt for refining a previous solution."""
        return f"""Here is a math problem and a previous attempt at solving it.
Please carefully check the solution for errors, then provide an improved solution.
Put your final answer in \\boxed{{}}.

Problem: {problem}

Previous attempt:
{previous}

Improved solution:"""

    def _generate(self, prompt: str) -> dict:
        """Internal generation with token counting."""
        if self.use_vllm:
            return self._generate_vllm(prompt)
        else:
            return self._generate_hf(prompt)

    def _generate_vllm(self, prompt: str) -> dict:
        """Generate using vLLM."""
        outputs = self.model.generate([prompt], self.sampling_params)
        output = outputs[0]
        response = output.outputs[0].text
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _generate_hf(self, prompt: str) -> dict:
        """Generate using HuggingFace transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_tokens = outputs.shape[1] - input_tokens
        response = self.tokenizer.decode(
            outputs[0][input_tokens:], skip_special_tokens=True
        )

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def generate_batch(self, prompts: list[str]) -> list[dict]:
        """Batch generation for efficiency."""
        if self.use_vllm:
            return self._generate_batch_vllm(prompts)
        else:
            return self._generate_batch_hf(prompts)

    def _generate_batch_vllm(self, prompts: list[str]) -> list[dict]:
        """Batch generate using vLLM."""
        outputs = self.model.generate(prompts, self.sampling_params)
        results = []
        for out in outputs:
            results.append({
                "response": out.outputs[0].text,
                "input_tokens": len(out.prompt_token_ids),
                "output_tokens": len(out.outputs[0].token_ids),
            })
        return results

    def _generate_batch_hf(self, prompts: list[str]) -> list[dict]:
        """Batch generate using HuggingFace (falls back to sequential)."""
        # HuggingFace batching is more complex, fall back to sequential
        return [self._generate_hf(p) for p in prompts]


def load_model(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> ModelWrapper:
    """Load model, trying vLLM first."""
    try:
        return ModelWrapper(
            model_name,
            use_vllm=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ImportError:
        print("vLLM not available, using HuggingFace (slower)", file=sys.stderr)
        return ModelWrapper(
            model_name,
            use_vllm=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"vLLM failed ({e}), falling back to HuggingFace", file=sys.stderr)
        return ModelWrapper(
            model_name,
            use_vllm=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    # Quick test
    print("Testing model loading...")
    model = load_model("Qwen/Qwen2.5-0.5B-Instruct")  # Small model for testing

    print("\nTesting single generation...")
    result = model.generate_solution("What is 2 + 2?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")

    print("\nTesting batch generation...")
    prompts = [model._make_solution_prompt("What is 1 + 1?")] * 3
    results = model.generate_batch(prompts)
    print(f"Generated {len(results)} responses")
