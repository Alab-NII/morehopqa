import random
random.seed(42)

QUESTION_PROMPT = """
Answer the following question:
#QUESTION\n
"""

CONTEXT_ADDITION = """Context: #CONTEXT\n"""

FEWSHOT_ADDITION = """
Here are some example questions:

#FEWSHOT\n

These were all examples.
"""

FORMAT_PROMPT = """
You will be asked a question, and should provide a short answer.
If the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)
If the answer is a name, format it as follows: Firstname Lastname
If the answer contains any number, format it as a number, not a word, and only output that number. Do not include leading 0s.

Please provide the answer in the following format: <answer>your answer here</answer>
Answer as short as possible.\n
"""

class PromptGenerator:
    
    @staticmethod
    def create(prompt_type, dataset=None):
        if prompt_type == "zeroshot":
            return ZeroShotGenerator()
        elif prompt_type == "2-shot":
            return FewShotGenerator(dataset, 2)
        elif prompt_type == "3-shot":
            return FewShotGenerator(dataset, 3)
        elif prompt_type == "zeroshot-cot":
            return ZeroShotGenerator(cot=True)
        elif prompt_type == "2-shot-cot":
            return FewShotGenerator(dataset, shots=2, cot=True)
        elif prompt_type == "3-shot-cot":
            return FewShotGenerator(dataset, shots=3, cot=True)
        

class ZeroShotGenerator(PromptGenerator):
    def __init__(self, cot=False):
        self.cot = cot
        
    def get_prompt(self, question_entry, context, question):
        prompt =  FORMAT_PROMPT + "\n"
        if context is not None:
            prompt += "\n" + CONTEXT_ADDITION
            context_string = ""
            for i in range(len(context)):
                context_paragraph = context[i]
                context_string += "\n" + f"{i+1}: " + context_paragraph[0] + "\n" + " ".join(context_paragraph[1])
            prompt = prompt.replace("#CONTEXT", context_string)
        prompt += QUESTION_PROMPT.replace("#QUESTION", question) + "\n"
        if self.cot:
            prompt += "Let's think step by step."
        
        return prompt
    

class FewShotGenerator(PromptGenerator):
    def __init__(self, dataset, shots=2, cot=False):
        self.shots = shots
        self.dataset = dataset
        self.cot = cot

    def get_fewshot_examples(self, question_entry, question):
        possible_entries = [entry for entry in self.dataset.items() if (entry['answer_type'] == question_entry['answer_type']) and (entry['previous_answer_type'] == question_entry['previous_answer_type']) and (entry['_id'].split("_")[:-1] != question_entry['_id'].split("_")[:-1]) and (entry['_id'].split("_")[-1] != question_entry['_id'].split("_")[-1])]
        fewshot_entries = random.sample(possible_entries, self.shots) if len(possible_entries) >= self.shots else possible_entries

        res = ""
        for j in range(len(fewshot_entries)):
            fewshot_entry = fewshot_entries[j]
            res += f"\nThis is example {j+1}: \n"
            if self.cot:
                context = fewshot_entry["context"]
                res += CONTEXT_ADDITION
                context_string = ""
                for i in range(len(context)):
                    context_paragraph = context[i]
                    context_string += "\n" + f"{i+1}: " + context_paragraph[0] + "\n" + " ".join(context_paragraph[1])
                res = res.replace("#CONTEXT", context_string)

                if question_entry['question'] == question:
                    res += f"""\nQuestion: {fewshot_entry['question']}\n"""
                elif question_entry['previous_question'] == question:
                    res += f"""\nQuestion: {fewshot_entry['previous_question']}\n"""
                elif question_entry['ques_on_last_hop'] == question:
                    res += f"""\nQuestion: {fewshot_entry['ques_on_last_hop']}\n"""
                elif question_entry['question_decomposition'][0]["question"] == question:
                    res += f"""\nQuestion: {fewshot_entry['question_decomposition'][0]["question"]}\n"""
                elif question_entry['question_decomposition'][1]["question"] == question:
                    res += f"""\nQuestion: {fewshot_entry['question_decomposition'][1]["question"]}\n"""
                elif question_entry['question_decomposition'][2]["question"] == question:
                    res += f"""\nQuestion: {fewshot_entry['question_decomposition'][2]["question"]}\n"""
                else:
                    raise ValueError(f"Something changed with this question: {question}")

                subquestions = fewshot_entry["question_decomposition"]
                if question_entry['question'] == question:
                    pass
                elif question_entry['previous_question'] == question:
                    subquestions = subquestions[:2]
                elif question_entry['ques_on_last_hop'] == question:
                    subquestions = subquestions[1:]
                elif question_entry['question_decomposition'][0]["question"] == question:
                    subquestions = []
                elif question_entry['question_decomposition'][1]["question"] == question:
                    subquestions = []
                elif question_entry['question_decomposition'][2]["question"] == question:
                    subquestions = [subquestions[2]]
                else:
                    raise ValueError(f"Something changed with this question: {question}")
                
                res += "\nAnswer: "
                if subquestions:
                    res += "Let's split the question into subquestions: \n"
                for subquestion in subquestions:
                    if "details" in subquestion.keys():
                        for detail in subquestion["details"]:
                            res += f"- {detail['question']} {detail['answer']}\n"
                    else:
                        res += f"- {subquestion['question']} {subquestion['answer']}\n"

                if subquestions:
                    res += "Therefore, the final answer is: "
                else:
                    res += "The final answer is: "
                if question_entry['question'] == question:
                    res += f"<answer>{fewshot_entry['answer']}</answer>\n"
                elif question_entry['previous_question'] == question:
                    res += f"""<answer>{fewshot_entry['previous_answer']}</answer>\n"""
                elif question_entry['ques_on_last_hop'] == question:
                    res += f"""<answer>{fewshot_entry['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][0]["question"] == question:
                    res += f"""<answer>{fewshot_entry['question_decomposition'][0]['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][1]["question"] == question:
                    res += f"""<answer>{fewshot_entry['question_decomposition'][1]['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][2]["question"] == question:
                    res += f"""<answer>{fewshot_entry['question_decomposition'][2]['answer']}</answer>\n"""
                else:
                    raise ValueError(f"Something changed with this question: {question}")
            else:
                if question_entry['question'] == question:
                    res += f"""{CONTEXT_ADDITION}\n\nQuestion: {fewshot_entry['question']}\nThe final answer is: <answer>{fewshot_entry['answer']}</answer>\n"""
                elif question_entry['previous_question'] == question:
                    res += f"""{CONTEXT_ADDITION}\n\nQuestion: {fewshot_entry['previous_question']}\nThe final answer is:<answer>{fewshot_entry['previous_answer']}</answer>\n"""
                elif question_entry['ques_on_last_hop'] == question:
                    res += f"""{CONTEXT_ADDITION}\n\nQuestion: {fewshot_entry['ques_on_last_hop']}\nThe final answer is: <answer>{fewshot_entry['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][0]["question"] == question:
                    res += f"""{CONTEXT_ADDITION}\n\nQuestion: {fewshot_entry['question_decomposition'][0]["question"]}\nThe final answer is: <answer>{fewshot_entry['question_decomposition'][0]['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][1]["question"] == question:
                    res += f"""{CONTEXT_ADDITION}\n\nQuestion: {fewshot_entry['question_decomposition'][1]["question"]}\nThe final answer is: <answer>{fewshot_entry['question_decomposition'][1]['answer']}</answer>\n"""
                elif question_entry['question_decomposition'][2]["question"] == question:
                    res += f"""{fewshot_entry['question_decomposition'][2]["question"]}\nThe final answer is: <answer>{fewshot_entry['question_decomposition'][2]['answer']}</answer>\n"""
                else:
                    raise ValueError(f"Something changed with this question: {question}")
                context = fewshot_entry["context"]
                res += "\n"
                context_string = ""
                for i in range(len(context)):
                    context_paragraph = context[i]
                    context_string += "\n" + f"{i+1}: " + context_paragraph[0] + "\n" + " ".join(context_paragraph[1])
                res = res.replace("#CONTEXT", context_string)
            res += "\n\n"
        return res
   
    def get_prompt(self, question_entry, context, question):
        prompt =  FORMAT_PROMPT + "\n"
        prompt += FEWSHOT_ADDITION.replace("#FEWSHOT", self.get_fewshot_examples(question_entry, question))
        if context is not None:
            prompt += "\n" + CONTEXT_ADDITION
            context_string = ""
            for i in range(len(context)):
                context_paragraph = context[i]
                context_string += "\n" + f"{i+1}: " + context_paragraph[0] + "\n" + " ".join(context_paragraph[1])
            prompt = prompt.replace("#CONTEXT", context_string)
        prompt += QUESTION_PROMPT.replace("#QUESTION", question) + "\n"
        
        return prompt







