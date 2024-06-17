"""
Use OpenAI models to answer questions directly, i.e. prompt question by question instead of using the batch API.
This method is faster, but also more expensive.
"""

from models.abstract_model import AbstractModel
from openai import OpenAI
import json
from datetime import datetime
from tqdm import tqdm

SYSTEM_PROMPT = """
You are a question answering system. The user will ask you a question and you will provide an answer.
You can generate as much text as you want to get to the solution. Your final answer must be contained in two brackets: <answer> </answer>.
"""

OPENAI_PROMPT = """
Please answer the following question:

#QUESTION
"""

CONTEXT_ADDITION = """

The following context information will be useful to answer the question:

#CONTEXT
"""

END_OF_PROMPT = """
If the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)
If the answer is a name, format it as follows: Firstname Lastname
If the question is a yes or no question: answer with 'yes' or 'no' (without quotes)
If the answer contains any number, format it as a number, not a word, and only output that number.

Please provide the answer in the following format: <answer>*your answer here*</answer>
Answer as short as possible.
"""

class OpenAIDirectModel(AbstractModel):
    def __init__(self, model_name="gpt-3.5-turbo", output_file_name="output", prompt_generator=None):
        self.model = OpenAI()
        self.model_name = model_name.replace("-direct", "")
        self.output_file_name =  output_file_name
        self.prompt_generator = prompt_generator

    def generate_text(self, prompt, max_tokens=256):
        return self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        ).choices[0].message.content
    
    def get_prompt(self, question_entry, context, question):
        return self.prompt_generator.get_prompt(question_entry, context, question)

    def get_answer(self, prompt):
        return self.generate_text(prompt)
    
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

    def get_answers_and_cache(self, dataset):
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
    
def main():
    model = OpenAIDirectModel()
    question = "What is the capital of France?"
    context = [("France is a country in Europe.", ["It is known for its wine and cheese."])]
    print(model.get_prompt(question, context))
    answer = model.get_answer(question, context)
    print(answer)

if __name__ == "__main__":
    main()