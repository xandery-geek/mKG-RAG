# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools to evaluate answers in Encyclopedic-VQA."""

import re
import string
import functools
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_MODEL_PATH = "kortukov/answer-equivalence-bem"
_PUNCTUATION_CHARACTERS = string.punctuation + "‘’´`_"
_QUESTION_TYPES = ["templated", "automatic", "multi_answer", "2_hop"]
_DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def preprocess_answer(
    answer,
    punctuation_characters=_PUNCTUATION_CHARACTERS,
    replacement_character="",
):
    """Function to preprocess VQA answers."""

    def remove_articles(s):
        """Remove common articles and prefixes in the answer."""
        return re.sub(r"\b(the answer is|a|an|the)\b", " ", s)

    def replace_punctuation(s):
        """Replace punctuation characters."""
        to_replace = set(punctuation_characters)
        return "".join(replacement_character if c in to_replace else c for c in s)

    def white_space_fix(s):
        """Remove superfluous whitespace."""
        return " ".join(s.split())

    def remove_llm_span_prefix(answer, prefix="<extra_id_0> "):
        """Remove span prefix added by some LLM."""
        if answer.startswith(prefix):
            return answer.replace(prefix, replacement_character)
        return answer

    def standarize_digits_and_contractions(s):
        """Standarize the representation of some digits and common contractions."""
        output = []
        tmp = s.split()
        for w in tmp:
            w = _DIGIT_MAP.get(w, w)
            w = _CONTRACTIONS.get(w, w)
            output.append(w)
        return " ".join(output)

    answer = answer.lower().replace("\n", " ").replace("\t", " ").strip()
    answer = remove_llm_span_prefix(answer)
    answer = replace_punctuation(answer)
    answer = remove_articles(answer)
    answer = standarize_digits_and_contractions(answer)
    answer = white_space_fix(answer)

    return answer


def singleanswer_exact_match(reference, candidate):
    """Compute exact match between single reference and candidate answers."""
    preprocessed_reference = preprocess_answer(reference)
    preprocessed_candidate = preprocess_answer(candidate)
    if not preprocessed_reference:
        raise ValueError("Reference answer is empty after preprocessing.")
    return preprocessed_reference == preprocessed_candidate


def _list_intersection_over_union(target_list, prediction_list):
    """Computes intersection over the union for lists for multi-answer questions.

    Precondition: the target list is not empty.

    Args:
      target_list: List with the reference answers.
      prediction_list: List with the candidate answers.

    Returns:
      A boolean indicating if the lists are an exact match.

    Raises:
      ValueError: If target_list is empty.
    """
    if not target_list:
        raise ValueError("Target list should not be empty.")
    target_set = set(target_list)
    prediction_set = set(prediction_list)
    intersection = target_set.intersection(prediction_set)
    union = target_set.union(prediction_set)
    return len(intersection) / len(union)


def multianswer_exact_match(reference, candidate, iou_threshold=0.5):
    """Computes an exact match score for multi_answer questions."""
    reference_list = reference.split("&&")
    reference_list = [preprocess_answer(a) for a in reference_list]
    reference_list = [a for a in reference_list if a]
    if not reference_list:
        raise ValueError("Reference list is empty after preprocessing.")
    candidate_list = candidate.replace(" and ", ",").replace(" & ", ",").split(",")
    candidate_list = [preprocess_answer(a) for a in candidate_list]
    candidate_list = [a for a in candidate_list if a]
    iou = _list_intersection_over_union(reference_list, candidate_list)
    return iou >= iou_threshold


def exact_match_scoring_function(example):
    """Score an example using exact match (EM)."""
    if example["question_type"] == "multi_answer":
        return multianswer_exact_match(example["reference"], example["candidate"])
    return singleanswer_exact_match(example["reference"], example["candidate"])


def initialize_bem_scoring_function(model_path, device="cpu"):
    """Instantiates and returns a function to compute BEM scores in PyTorch."""

    # 1 - Get BEM tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bem = AutoModelForSequenceClassification.from_pretrained(model_path)
    bem.to(device)
    bem.eval()

    # 2 - Define helper functions
    def tokenize_function(question, reference, candidate):
        text = f"[CLS] {candidate} [SEP]"
        text_pair = f"{reference} [SEP] {question} [SEP]"
        return tokenizer(text=text, 
                         text_pair=text_pair, 
                         add_special_tokens=False, 
                         padding='max_length',
                         max_length=512,
                         truncation=True, 
                         return_tensors='pt')

    def score_example(
        example,
        threshold_score=True,
    ):
        """Scores an Encyclopedic-VQA example using BEM.

        The BEM evaluation function uses a BERT model trained to score answer
        equivalence and is described in https://arxiv.org/abs/2202.07654.

        Args:
          example: Dict containing the example to score.
          threshold_score: Whether to threshold the score (>= 0.5) to a boolean.

        Returns:
          Score based on BEM for encyclopedic-VQA examples.

        Raises:
          ValueError: if reference answer is empty.
        """
        if not example["reference"]:
            raise ValueError("Reference answer cannot be empty.")

        # Preprocess list questions
        if example["question_type"] in ["list", "multianswer", "multi_answer"]:
            example["reference"] = example["reference"].replace("&&", ",")

        inputs = tokenize_function(example["question"], example["reference"], example["candidate"])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = bem(**inputs)
        score = F.softmax(out.logits, dim=-1).argmax().item()
        
        if threshold_score:
            return float(score >= 0.5)
        return score

    return score_example


def encyclopedic_vqa_evaluation_function(example, bem_scoring_function):
    """Scores an example using the Encyclopedic-VQA evaluation function.

    It evaluates the example using Exact Match, and if the result is negative,
    then it uses BEM (described in https://arxiv.org/abs/2202.07654).
    Note that for single-answer questions, Exact Match is always stricter than
    BEM. Therefore, skipping Exact Match and using only BEM would yield the same
    result, but it would be much slower to compute.

    Args:
      example: Example to evaluate.
      bem_scoring_function: function to compute BEM scores.

    Returns:
      Encyclopedic-VQA score.

    Raises:
      ValueError: if reference answer is empty or question_type is incorrect.
    """
    if not example["reference"]:
        raise ValueError("Reference answer cannot be empty.")
    if example["question_type"] not in _QUESTION_TYPES:
        raise ValueError(f"Unknown question type. Valid options are {_QUESTION_TYPES}")
    matches_exactly = exact_match_scoring_function(example)
    if matches_exactly:
        return 1.0
    return bem_scoring_function(example, threshold_score=True)


@functools.cache
def initialize_encyclopedic_vqa_evaluation_function(
    model_path,
    device="cpu",
):
    """Instantiates and returns a function to compute Encyclopedic-VQA scores."""
    bem_scoring_function = initialize_bem_scoring_function(model_path=model_path, device=device)
    return functools.partial(
        encyclopedic_vqa_evaluation_function,
        bem_scoring_function=bem_scoring_function,
    )


def evaluate_example(
    example,
    device="cpu",
):
    """Prepares and evaluates examples with the Encyclopedic-VQA function.

    Args:
      question: Text of the question to evaluate.
      reference_list: List of ground truth reference answers.
      candidate: Candidate answer to evaluate.
      question_type: Indicates the type of question to evaluate.

    Returns:
      The maximum score obtained by evaluating the candidate answer against each
      possible reference answer for a given question.
    """
    if not example["reference_list"]:
        raise ValueError("Reference list cannot be empty.")

    scoring_function = initialize_encyclopedic_vqa_evaluation_function(_MODEL_PATH, device=device)

    scores = []
    for reference in example["reference_list"]:
        score = scoring_function({
            "question": example["question"],
            "reference": reference,
            "candidate": example["candidate"],
            "question_type": example["question_type"],
        })
        scores.append(score)
    score = max(scores)
    return {example["question_id"]: score}
