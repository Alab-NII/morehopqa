"""
Implement wrapper for Llama-70b.
"""

from models.abstract_model import AbstractModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import torch

SYSTEM_PROMPT = """
You are a question answering system. The user will ask you a question and you will provide an answer.
You can generate as much text as you want to get to the solution. Your final answer must be contained in two brackets: <answer> </answer>.
"""

class Llama70b(AbstractModel):

    def __init__(self, model_name="llama-70b", output_file_name="output", prompt_generator=None):
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
        self.model_name = model_name
        self.output_file_name =  output_file_name
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        self.prompt_generator = prompt_generator

    def get_prompt(self, question_entry, context, question):
        prompt = self.prompt_generator.get_prompt(question_entry, context, question)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        return chat
    
    def get_answer(self, prompt):
        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.model.generate(input_ids, max_new_tokens=256, do_sample=True, eos_token_id=terminators)
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
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