"""
This file contains the evaluation functions for the model. 
Mainly it compares the model's answer to the ground truth answer.
"""
import re
import spacy
import string
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def evaluate(answer):
    """Find correct answer. 
    Should go in two steps:
    1. Check content between <answer> and </answer> tags.
    2. If not there, take last occurrence of the right entity.
    """
    results = dict()
    for case_id in ["case_1", "case_2", "case_3", "case_4", "case_5", "case_6"]:
        model_answer_text = answer[case_id + "_pred_extr"]
        ground_truth_answer_text = answer[case_id + "_ground_truth"]
        results[case_id + "_em"] = exact_match_score(model_answer_text, ground_truth_answer_text)
        results[case_id + "_f1"], results[case_id + "_precision"], results[case_id + "_recall"] = f1_score(model_answer_text, ground_truth_answer_text)
    return results


def evaluate_baseline(answers: dict):
    """Evaluate the baseline model. Ignores cases 3 to 6."""
    res = dict()
    total_answers = 0
    data = answers.values()
    for entry in tqdm(data, total=len(answers)):
        result_entry = deepcopy(entry)
        total_answers += 1
        result_entry["_id"] = entry["_id"]
        for case_id in ["case_1", "case_2"]:
            model_answer_text = entry[case_id + "_pred_extr"]
            ground_truth_answer_text = entry[case_id + "_ground_truth"]
            result_entry[case_id + "_em"] = exact_match_score(model_answer_text, ground_truth_answer_text)
            result_entry[case_id + "_f1"], result_entry[case_id + "_precision"], result_entry[case_id + "_recall"] = f1_score(model_answer_text, ground_truth_answer_text)
        res[result_entry["_id"]] = result_entry
    return res


def evaluate_all(answers: dict):
    """Evaluate all answers from the model and compare them to the ground truth."""
    res = dict()
    total_answers = 0
    for entry in tqdm(answers.values(), total=len(answers)):
        result_entry = deepcopy(entry)
        total_answers += 1
        result_entry.update(evaluate(result_entry))
        res[result_entry["_id"]] = result_entry
    return res
