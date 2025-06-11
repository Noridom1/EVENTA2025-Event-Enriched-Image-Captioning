from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Summarizer:
    def __init__(self, model_name="Qwen/Qwen3-14B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded successfully.")

    def format_prompt(self, article_text):
        return f"Please summarize the following article:\n\n{article_text}"

    def summarize(self, article_text, max_new_tokens=1024):
        if not self.tokenizer or not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")

        prompt = self.format_prompt(article_text)
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Try to extract thinking content if available
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        summary_content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return summary_content
