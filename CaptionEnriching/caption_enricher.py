from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CaptionEnricher:
    def __init__(self, model_name="Qwen/Qwen3-4B", max_new_token=5000):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_new_token = max_new_token

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def format_prompt(self, query):
        prompt = (
            "Create a detail caption by combining both the image caption and the information of article I provide. "
            "Only return plain text with no formatting:"
            + "\nThe image description:\n" + query["image_raw_caption"] 
            + "\nThe image caption from the article: " + query["web_caption"]
            + "\nThe summary of article:" 
            + "\nTitle: " + query["title"]
            + "\n" + query["summary_article_content"]
        )
        return prompt
    
    def get_sys_prompt(self):
        sys_prompt = (
            "You are an expert journalistic assistant tasked with creating detailed and informative image captions.\n"
            "Your goal is to combine information from image descriptions and the information of an article "
            "(from sources like The Guardian or CNN) into a single, enriched caption. \n"
            "The image sometimes seem irrelevant to the article, but try to the context given by the article to generate an appropriate caption.\n"
            "Example: an article talking about a tennis event in the Olympic, but the photo is about the swimming event in such Olympic. "
            "You need to take the scenario of the Olympic to generate the caption for the image relating to swimming.\n"
            "The output must be plain text without any formatting. The language of the caption MUST BE in English.\n"
            "The number of words of a caption should be from 100 to 140 words."
        )
        return sys_prompt

    def get_sys_prompt_chunk(self):
        sys_prompt = """
            You are an expert journalistic assistant tasked with creating detailed and informative image captions. 
            Your goal is to combine information from a image descriptions and the information of a article 
            (from sources The Guardian or CNN) into a single, enriched caption.

            Instructions:
            1. Identify any generic reference in the base caption (e.g. “a person,” “the driver,” “the protester”).
            2. Extract named entities from the article (person names, locations, date, figures and statistics, etc.)
            3. Match the corresponding named entity (person or object) in the article chunks into the generic reference, and form a coherent, single enriched caption.
            4. The named-entities in the article should takes precedence over the image description named-entities, i.e. the named-entities in the image description may be incorrect

            Requirements:
            1. Output in plain text without any formatting.
            2. Use English only. Do not put any Chinese or other languages' characters into the final caption.
        """
        return sys_prompt

    def enrich_caption(self, query):
        if not self.tokenizer or not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")

        sys_prompt = self.get_sys_prompt()
        prompt = self.format_prompt(query)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_token,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Try to extract thinking section (if Qwen uses </think>)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # </think> token
        except ValueError:
            index = 0

        enriched_caption = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return enriched_caption
    
    def enrich_caption_batch(self, queries):
        if not self.tokenizer or not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")

        sys_prompt = self.get_sys_prompt()

        texts = []
        for query in queries:
            prompt = self.format_prompt(query)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(chat_text)

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_token,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            # repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_lens = [len(input_ids) for input_ids in model_inputs["input_ids"]]
        outputs = []

        for i, gen_ids in enumerate(generated_ids):
            output_ids = gen_ids[input_lens[i]:].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                index = 0
            caption = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            outputs.append(caption)

        return outputs
    
    def enrich_caption_chunk_batch(self, queries):
        if not self.tokenizer or not self.model:
            raise ValueError("Model is not loaded. Call `load_model()` first.")

        sys_prompt = self.get_sys_prompt_chunk()

        texts = []
        for query in queries:
            prompt = self.format_prompt(query)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(chat_text)

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_token,
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
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                index = 0
            caption = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            outputs.append(caption)

        return outputs
