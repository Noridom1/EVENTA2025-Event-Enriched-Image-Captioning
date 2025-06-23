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
        return  f"Summarize this article in English to capture key but general information about the event in the article." \
                f"Summarize the article as detailed as possible to cover all possible notable events in the article.\n" \
                f"Keep the number of words of the summarized article between 150-300 words" \
                f"Output in plain text without any formatting:\n\n{article_text}"

    def summarize(self, article_text, max_new_tokens=5000):
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
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
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
    
    def summarize_batch(self, article_texts, max_new_tokens=5000):
        if not self.tokenizer or not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")

        prompts = [self.format_prompt(text) for text in article_texts]

        messages_batch = [
            [{"role": "user", "content": prompt}] for prompt in prompts
        ]

        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            ) for messages in messages_batch
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_lens = [len(input_ids) for input_ids in model_inputs["input_ids"]]
        outputs = []

        for i, gen_ids in enumerate(generated_ids):
            output_ids = gen_ids[input_lens[i]:].tolist()

            # Try to find end of thinking section if available
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                index = 0

            summary_content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")

            outputs.append(summary_content)

        return outputs
