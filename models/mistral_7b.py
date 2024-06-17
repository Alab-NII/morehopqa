"""
Implement wrapper for Mistral-7b.
"""

from models.abstract_model import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import torch

class Mistral7B(AbstractModel):

    def __init__(self, model_name="mistral-7b", output_file_name="output", prompt_generator=None):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map="cuda", torch_dtype=torch.bfloat16)
        self.model_name = model_name
        self.output_file_name =  output_file_name
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.prompt_generator = prompt_generator

    def get_prompt(self, question_entry, context, question):
        prompt = self.prompt_generator.get_prompt(question_entry, context, question)
        chat = [
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    def get_answer(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(**input_ids, max_new_tokens=256, do_sample=True)
        return self.tokenizer.decode(outputs[0])[len(prompt):]
    
    def get_all_cases(self, entry):
        cases = dict()
        context = entry["context"]
        cases["case_1"] = self.get_prompt(entry, context, entry['question'])
        cases["case_2"] = self.get_prompt(entry, context, entry['previous_question'])
        cases["case_3"] = self.get_prompt(entry, context, entry['ques_on_last_hop'])
        cases["case_6"] = self.get_prompt(entry, context, entry['question_decomposition'][0]["question"])
        cases["case_5"] = self.get_prompt(entry, context, entry['question_decomposition'][1]["question"])
        cases["case_4"] = self.get_prompt(entry, None, entry['question_decomposition'][2]["question"])

        return cases


    def get_answers_and_cache(self, dataset) -> dict:
        answers = dict()
        for entry in tqdm(dataset.items(), total=dataset.length):
            cases = self.get_all_cases(entry)
            answer_entry = dict()
            answer_entry["_id"] = entry["_id"]
            answer_entry["context"] = entry["context"]
            for case_id, prompt in cases.items():
                answer = self.get_answer(prompt)
                answer_entry[f"{case_id}_prompt"] = prompt
                answer_entry[f"{case_id}_answer"] = answer
                
            answers[entry["_id"]] = answer_entry
            with open(f"models/cached_answers/{self.output_file_name}", "w") as f:
                json.dump(answers, f, indent=4)

        return answers