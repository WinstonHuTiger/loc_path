import argparse
import ast
import csv
import json
import os
import re
import sys
import time
import random
import hashlib
from tqdm import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None
try:
    from nltk.translate.bleu_score import sentence_bleu
except ImportError:  # pragma: no cover - optional dependency
    sentence_bleu = None
try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:  # pragma: no cover - optional dependency
    meteor_score = None
try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None

# ----------------------------
# Prompts (same as before)
# ----------------------------
wsi_relevance_scoring = """
System Message: Please act as an impartial judge and evaluate the relevance of the original ground truth answer to each claim derived from the model’s answer. Provide an explanation for each evaluation and assign a score based on the following criteria.

**Scoring Criteria:**
- **1**: The content in the ground truth answer is completely relevant to the claim.
- **0.7**: The content is mostly relevant but has minor omissions or deviations.
- **0.3**: The content is partially relevant with significant omissions or irrelevant information.
- **0**: The content in the ground truth answer is not relevant to the claim.

Output Requirements:

Please output your evaluations as a list of dictionaries in plain text format (not JSON). The format should be as follows:

[
{
"claim": "Original claim1",
"explanation": "Explanation for the score",
"score": 1 or 0.7 or 0.3 or 0
},
{
"claim": "Original claim2",
"explanation": "Explanation for the score",
"score": 1 or 0.7 or 0.3 or 0
},
...
]
"""

claims_extraction_prompt = """
System Message: You are an AI assistant specialized in processing pathological diagnosis Q&A pairs. I will provide you with a pathology diagnosis question and its corresponding answer.
Your task is to:
Claims Extraction:
• Carefully analyze the answer and remove any unnecessary information that is not directly relevant to the question.
• Only extract claims that directly address the question. Discard any information that does not directly answer or pertain to the question.
• Divide the refined answer into several distinct and granular claims.
• Keep closely related information together in the same claim to preserve context and meaning. Do not split sentences or ideas that are logically connected.
• Break down complex sentences into smaller, individual claims only if it does not disrupt the logical flow or separate connected ideas.
• Ensure there is no omission or repetition among the claims.

Guidelines:
• Only output the claims without including any additional text or explanations.
• Each claim should be concise and represent a single fact or point directly related to the question.
• Maintain the integrity of statements that are contextually connected.
Output Format: Present the extracted claims as a list in the following format:
["claim1", "claim2", "claim3", ...]
"""

wsi_precision_scoring_prompt = """
System Message: Please act as an impartial judge and evaluate the correctness of the AI assistant's pathology dialogue for each claim based on the following scoring criteria. Provide an explanation for each evaluation and assign a score.

**Scoring Criteria:**
- **1**: The information in the pathology dialogue is completely correct regarding the claim.
- **0.7**: The information is mostly correct and closely aligns with the claim.
- **0.3**: The claim is mentioned but contains errors in the core content (e.g., mistakes in differentiation degree or malignancy).
- **0**: The information in the pathology dialogue is completely incorrect regarding the claim.

Output Requirements:
Please output your evaluations as a list of dictionaries in plain text format (not JSON). The format should be as follows:

[
{
"claim": "Original claim1",
"explanation": "Explanation for the score",
"score": 1 or 0.7 or 0.3 or 0
},
{
"claim": "Original claim2",
"explanation": "Explanation for the score",
"score": 1 or 0.7 or 0.3 or 0
},
...
]
"""

# ----------------------------
# Report metric helpers
# ----------------------------

REPORT_BLEU_WEIGHTS = {
    "BLEU-1": (1.0, 0.0, 0.0, 0.0),
    "BLEU-2": (0.5, 0.5, 0.0, 0.0),
    "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0.0),
    "BLEU-4": (0.25, 0.25, 0.25, 0.25),
}

def _ensure_report_metric_deps():
    missing = []
    if np is None:
        missing.append("numpy")
    if sentence_bleu is None or meteor_score is None:
        missing.append("nltk")
    if rouge_scorer is None:
        missing.append("rouge-score")
    if missing:
        raise RuntimeError(
            "Missing optional dependencies for report metrics: " + ", ".join(missing)
        )

def calculate_report_nlu_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-1/2/3/4, ROUGE-L, and METEOR following the reference NLU logic.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis lists must have identical lengths.")
    if not references:
        metric_keys = list(REPORT_BLEU_WEIGHTS.keys()) + ["ROUGE-L", "METEOR"]
        return {k: 0.0 for k in metric_keys}

    _ensure_report_metric_deps()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    metric_keys = list(REPORT_BLEU_WEIGHTS.keys()) + ["ROUGE-L", "METEOR"]
    score_lists: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for ref, hyp in zip(references, hypotheses):
        ref_text = str(ref or "")
        hyp_text = str(hyp or "")
        ref_token_list = ref_text.split()
        ref_tokens = [ref_token_list]
        hyp_tokens = hyp_text.split()

        for name, weights in REPORT_BLEU_WEIGHTS.items():
            try:
                bleu_score = float(sentence_bleu(ref_tokens, hyp_tokens, weights=weights))
            except ZeroDivisionError:
                bleu_score = 0.0
            score_lists[name].append(max(bleu_score, 0.0))

        rouge_scores = scorer.score(ref_text, hyp_text)
        rouge_l = float(rouge_scores["rougeL"].fmeasure)
        score_lists["ROUGE-L"].append(max(rouge_l, 0.0))

        try:
            meteor_val = float(meteor_score([ref_token_list], hyp_tokens))
        except LookupError as exc:
            raise RuntimeError(
                "METEOR needs the NLTK WordNet data; run: "
                "python -m nltk.downloader wordnet omw-1.4") from exc
        except Exception:
            meteor_val = 0.0
        score_lists["METEOR"].append(max(meteor_val, 0.0))

    metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in score_lists.items()}
    return metrics

# ----------------------------
# OpenAI-compatible Chat client
# ----------------------------

@dataclass
class LLMConfig:
    api_base: str
    api_key: str
    model: str
    timeout_s: int = 120
    max_retries: int = 3
    temperature: float = 0.0
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    min_p: Optional[float] = 0.0
    provider: str = "openai-compatible"  # or "openai"

class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, messages: List[Dict[str, str]], **extra) -> str:
        url = self.cfg.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            **extra
        }
        # Attach sampling params only for openai-compatible backends
        if self.cfg.provider == "openai-compatible":
        
            if self.cfg.top_p is not None:
                # print('add top_p to the payload')
                payload["top_p"] = self.cfg.top_p
            if self.cfg.top_k is not None:
                payload["top_k"] = self.cfg.top_k
            if self.cfg.min_p is not None:
                payload["min_p"] = self.cfg.min_p
        
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.session.post(url, data=json.dumps(payload), timeout=self.cfg.timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    if attempt == self.cfg.max_retries:
                        raise RuntimeError(err)
                    time.sleep(1.5 * attempt)
            except Exception:
                if attempt == self.cfg.max_retries:
                    raise
                time.sleep(1.5 * attempt)
        raise RuntimeError("unreachable")

# ----------------------------
# Utilities
# ----------------------------

CHOICE_PATTERN = re.compile(r"\b[\(\[]?\s*([A-Ea-e])\s*[\)\]]?\b")
VALID_CHOICES = set(list("ABCDEabcde"))

YES_PAT = re.compile(r"\b(yes|y|true|t|correct|right|positive)\b", re.I)
NO_PAT  = re.compile(r"\b(no|n|false|f|incorrect|wrong|negative)\b", re.I)
CN_YES  = re.compile(r"[是对對真正确正]", re.U)
CN_NO   = re.compile(r"[非否错錯假不]", re.U)

def looks_like_mcq_letter(s: str) -> Optional[str]:
    if not s:
        return None
    stripped = str(s).strip()
    if len(stripped) == 1 and stripped in VALID_CHOICES:
        return stripped.upper()
    m = CHOICE_PATTERN.search(stripped)
    if m:
        return m.group(1).upper()
    if len(stripped) <= 3:
        letters = [ch for ch in stripped if ch.isalpha()]
        if len(letters) == 1 and letters[0] in VALID_CHOICES:
            return letters[0].upper()
    return None

def looks_like_yesno(s: str) -> Optional[str]:
    if not s:
        return None
    txt = str(s).strip()
    if YES_PAT.search(txt) or CN_YES.search(txt):
        return "YES"
    if NO_PAT.search(txt) or CN_NO.search(txt):
        return "NO"
    # very short tokens
    low = txt.lower()
    if low in {"yes", "y", "true", "t", "1"}:
        return "YES"
    if low in {"no", "n", "false", "f", "0"}:
        return "NO"
    return None

def extract_choice_from_model_output(s: str) -> Optional[str]:
    if not s:
        return None
    # first try letter
    m = CHOICE_PATTERN.search(s)
    if m:
        return m.group(1).upper()
    if len(s.strip()) <= 10:
        return looks_like_mcq_letter(s)
    m2 = re.search(r"(answer|final|prediction)\s*[:\-]\s*([A-Ea-e])\b", s, re.I)
    if m2:
        return m2.group(2).upper()
    return None

def extract_yesno_from_model_output(s: str) -> Optional[str]:
    if not s:
        return None
    txt = str(s)
    yn = looks_like_yesno(txt)
    if yn:
        return yn
    # handle short tokens / brackets
    short = re.sub(r"[^A-Za-z\u4e00-\u9fff]+", "", txt).lower()
    if short in {"yes", "true"}:
        return "YES"
    if short in {"no", "false"}:
        return "NO"
    return None

def tolerant_list_parser(text: str) -> List[Any]:
    if text is None:
        raise ValueError("No text to parse.")
    cleaned = text.strip()
    cleaned = re.sub(r"^```.*?\n", "", cleaned, flags=re.S)
    cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'")
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    lb = cleaned.find('[')
    rb = cleaned.rfind(']')
    if lb != -1 and rb != -1 and rb > lb:
        candidate = cleaned[lb:rb+1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        try:
            data = ast.literal_eval(candidate)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    try:
        data = ast.literal_eval(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    raise ValueError("Failed to parse list from text: " + cleaned[:200])

def parse_score_dicts(text: str) -> List[Dict[str, Any]]:
    raw_list = tolerant_list_parser(text)
    fixed: List[Dict[str, Any]] = []
    for item in raw_list:
        if isinstance(item, dict):
            claim = item.get("claim", "")
            explanation = item.get("explanation", "")
            score = item.get("score", 0)
            try:
                score = float(score)
            except Exception:
                m = re.search(r"(1(?:\.0)?|0\.7|0\.3|0(?:\.0)?)", str(score))
                score = float(m.group(1)) if m else 0.0
            fixed.append({"claim": str(claim), "explanation": str(explanation), "score": score})
        else:
            fixed.append({"claim": str(item), "explanation": "", "score": None})
    return fixed

# ----------------------------
# Report metrics via Hugging Face + sacrebleu
# ----------------------------

def compute_report_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Wrapper for single-sample report NLU metrics.
    """
    metrics = calculate_report_nlu_scores(
        [reference or ""],
        [hypothesis or ""],
    )
    return metrics

def compute_report_metrics_corpus(
    references: List[str],
    hypotheses: List[str],
) -> Optional[Dict[str, float]]:
    if not references or not hypotheses or len(references) != len(hypotheses):
        return None
    return calculate_report_nlu_scores(references, hypotheses)

# ----------------------------
# LLM Tasks
# ----------------------------

def llm_extract_claims(client: 'OpenAICompatClient', question: Optional[str], model_answer: str) -> List[str]:
    sys_msg = {"role": "system", "content": claims_extraction_prompt}
    usr_msg = {"role": "user", "content": json.dumps({"question": question or "", "answer": model_answer or ""}, ensure_ascii=False)}
    text = client.chat([sys_msg, usr_msg])
    try:
        claims = tolerant_list_parser(text)
        return [str(c).strip() for c in claims if str(c).strip()]
    except Exception:
        lines = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 1]

def llm_relevance_score(client: 'OpenAICompatClient', claims: List[str], ground_truth: str) -> List[Dict[str, Any]]:
    sys_msg = {"role": "system", "content": wsi_relevance_scoring}
    usr_msg = {"role": "user", "content": json.dumps({"claims": claims, "ground_truth_answer": ground_truth or ""}, ensure_ascii=False)}
    text = client.chat([sys_msg, usr_msg])
    return parse_score_dicts(text)

def llm_precision_score(client: 'OpenAICompatClient', claims: List[str], model_answer: str, ground_truth: str) -> List[Dict[str, Any]]:
    sys_msg = {"role": "system", "content": wsi_precision_scoring_prompt}
    usr_msg = {"role": "user", "content": json.dumps({"claims": claims, "model_answer": model_answer or "", "ground_truth_answer": ground_truth or ""}, ensure_ascii=False)}
    text = client.chat([sys_msg, usr_msg])
    return parse_score_dicts(text)

# ------- GT claims cache helpers -------
def _hash_text(txt: str) -> str:
    h = hashlib.sha256()
    h.update(txt.strip().encode("utf-8"))
    return h.hexdigest()

def load_cached_gt_claims(cache_dir: Path, gt_text: str) -> Optional[List[str]]:
    if not gt_text:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{_hash_text(gt_text)}.json"
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # print('cache hit!! for', gt_text)
                return [str(x) for x in data]
        except Exception:
            return None
    return None

def save_cached_gt_claims(cache_dir: Path, gt_text: str, claims: List[str]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = cache_dir / f"{_hash_text(gt_text)}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(list(claims), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ItemResult:
    idx: int
    kept: bool
    reason: str
    is_mcq: bool  # True for any close-type (MCQ or TF)
    mcq_ref: Optional[str] = None   # 'A'..'E' or 'YES'/'NO'
    mcq_pred: Optional[str] = None
    mcq_correct: Optional[bool] = None
    # Open-ended metrics
    gt_claims: Optional[List[str]] = None         # claims extracted from Ground Truth
    model_claims: Optional[List[str]] = None      # claims extracted from model output
    relevance: Optional[List[Dict[str, Any]]] = None  # per-claim scores (model_claims vs GT)
    precision: Optional[List[Dict[str, Any]]] = None  # per-claim scores (gt_claims vs model)
    wsi_r_final: Optional[float] = None           # average over relevance claim scores
    wsi_p_final: Optional[float] = None           # average over precision claim scores
    report_metrics: Optional[Dict[str, float]] = None # BLEU/ROUGE-L/METEOR for report generation

# ----------------------------
# File-based category classification (prefix-precise)
# ----------------------------

def _norm_stem(file_path: str) -> str:
    stem = Path(file_path).name
    stem = re.sub(r"\.jsonl?$", "", stem, flags=re.I)
    stem = stem.lstrip("_")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return stem

def _match_prefix(stem: str, prefix: str) -> bool:
    return stem == prefix or stem.startswith(prefix + "_")

def classify_task_by_file(file_path: str) -> Optional[str]:
    """
    Return one of:
      - "BCNB"
      - "PathBench"
      - "Morphological Analysis"
      - "Diagnosis"
      - "Treatment Planning"
      - "Report Generation"   <-- separated as requested
    """
    if not file_path:
        return None
    s = _norm_stem(file_path)

    if "bcnb" in s:
        return "BCNB"
    if "pathbench" in s:
        return "PathBench"

    # Morphological Analysis
    morph_prefixes = [
        "global_morphology_description",
        "key_diagnostic_description",
        "regional_structure_description",
        "specific_feature_description",
        "morphology_choice",
    ]
    for p in morph_prefixes:
        if _match_prefix(s, p):
            return "Morphological Analysis"

    # Report Generation (独立出来)
    report_prefixes = [
        "report",
        "report_generation",
        "reporting",
    ]
    for p in report_prefixes:
        if _match_prefix(s, p):
            return "Report Generation"

    # Diagnosis  （不再包含 'report'）
    diagnosis_prefixes = [
        "diagnosis_choice",
        "histological_typing",
        "grading",
        "molecular_subtyping_choice",
        "molecular_subtyping",
        "staging_choice",
        "staging",
    ]
    for p in diagnosis_prefixes:
        if _match_prefix(s, p):
            return "Diagnosis"

    # Treatment Planning
    plan_prefixes = [
        "treatment_planing",
        "treatment_planning",
        "prognosis_tf_question",
        "prognosis",
    ]
    for p in plan_prefixes:
        if _match_prefix(s, p):
            return "Treatment Planning"

    # fallbacks (rare)
    if any(k in s for k in ["report_generation", "reporting", "report"]):
        return "Report Generation"
    if any(k in s for k in ["prognosis", "treatment_planing", "treatment_planning"]):
        return "Treatment Planning"
    if any(k in s for k in [
        "global_morphology_description", "key_diagnostic_description",
        "regional_structure_description", "specific_feature_description",
        "morphology"
    ]):
        return "Morphological Analysis"
    if any(k in s for k in [
        "diagnosis_choice", "histological_typing", "grading",
        "molecular_subtyping", "staging"
    ]):
        return "Diagnosis"
    return None

def is_file_close_question(file_path: str) -> bool:
    """
    Decide whether a test json file is classification-like (Close) or Open.
    - choice or prognosis_tf_question -> Close
    Else -> Open
    """
    if not file_path:
        return False
    s = _norm_stem(file_path)
    close_prefixes = [
        "morphology_choice",
        "diagnosis_choice",
        "staging_choice",
        "molecular_subtyping_choice",
        "prognosis_tf_question",
        "bcnb_choice_test",
        "pathbench_choice_test"
    ]
    for p in close_prefixes:
        if _match_prefix(s, p):
            return True
    return False  # default Open

# ----------------------------
# Item evaluation
# ----------------------------

def _extract_ref_from_conversations(row: Dict[str, Any]) -> Optional[str]:
    conv = row.get("conversations")
    if not isinstance(conv, list):
        return None
    # find the last 'gpt' turn as reference (your TF files)
    ref = None
    for turn in conv:
        if isinstance(turn, dict) and str(turn.get("from", "")).lower() == "gpt":
            ref = turn.get("value")
    return ref

def _get_reference_text(row: Dict[str, Any]) -> str:
    reference = (
        row.get("reference")
        or row.get("ground_truth")
        or row.get("ref_answer")
        or row.get("answer")
    )
    if not reference:
        reference = _extract_ref_from_conversations(row)
    return str(reference or "")

def _get_model_output_text(row: Dict[str, Any]) -> str:
    return str(
        row.get("model_output")
        or row.get("output")
        or row.get("prediction")
        or row.get("llava_output")
        or ""
    )

def eval_item(
    idx: int,
    row: Dict[str, Any],
    client: Optional[OpenAICompatClient],
    is_close_question: bool,
    gt_claims_cache_dir: Optional[Path] = None,
    report_only_mode: bool = False,
    metrics_only_mode: bool = False,
) -> ItemResult:
    question = row.get("question") or row.get("prompt")
    model_out = _get_model_output_text(row)
    reference = _get_reference_text(row)

    file_name = str(row.get("file", ""))
    task_category = classify_task_by_file(file_name)
    report_task = (task_category == "Report Generation")

    if not model_out.strip():
        return ItemResult(idx=idx, kept=False, reason="empty_model_output", is_mcq=False)

    if report_only_mode and not report_task:
        return ItemResult(
            idx=idx,
            kept=False,
            reason="report_metric_mode_skip",
            is_mcq=is_close_question,
        )

    # print("question:", question)
    # print()
    # print('model output:  ', model_out)
    # print()
    # print('reference: ', reference)
    # --- Start Evaluation Logic ---
    if is_close_question:
        # Close-ended question logic (MCQ or TF)
        mcq_ref_letter = looks_like_mcq_letter(str(reference))
        if mcq_ref_letter:
            mcq_pred = extract_choice_from_model_output(str(model_out)) or ""
            correct = (mcq_pred == mcq_ref_letter)
            return ItemResult(idx=idx, kept=True, reason="mcq_compared", is_mcq=True,
                              mcq_ref=mcq_ref_letter, mcq_pred=mcq_pred, mcq_correct=bool(correct))

        # If not an MCQ letter, try to evaluate as TF (Yes/No)
        yn_ref = looks_like_yesno(str(reference))
        if yn_ref:
            yn_pred = extract_yesno_from_model_output(str(model_out)) or ""
            correct = (yn_pred == yn_ref)
            return ItemResult(idx=idx, kept=True, reason="tf_compared", is_mcq=True,
                              mcq_ref=yn_ref, mcq_pred=yn_pred, mcq_correct=bool(correct))

        # If it's a "close" file but the answer format is unrecognized, treat as incorrect
        return ItemResult(idx=idx, kept=True, reason="close_unrecognized_ref", is_mcq=True,
                          mcq_ref=str(reference)[:20], mcq_pred="", mcq_correct=False)
    else:
        # Open-ended question logic
        gt_text = str(reference or "")
        report_metrics: Optional[Dict[str, float]] = None
        if report_task:
            report_metrics = compute_report_metrics(gt_text, str(model_out))

        if metrics_only_mode:
            # report_task is guaranteed True here due to the early return above
            return ItemResult(
                idx=idx,
                kept=True,
                reason="report_metric_mode",
                is_mcq=False,
                report_metrics=report_metrics,
            )

        if client is None:
            return ItemResult(idx=idx, kept=True, reason="complex_but_no_client", is_mcq=False)

        # ---- Extract & cache GT claims ----
        gt_claims: List[str] = []
        if gt_claims_cache_dir is not None:
            cached = load_cached_gt_claims(gt_claims_cache_dir, gt_text)
            if cached:
                gt_claims = cached
        if not gt_claims:
            gt_claims = llm_extract_claims(client, question, gt_text)
            if gt_claims_cache_dir is not None and gt_claims:
                save_cached_gt_claims(gt_claims_cache_dir, gt_text, gt_claims)
        if not gt_claims:
            gt_claims = [gt_text.strip()[:3000]]

        # Extract model claims（用于 WSI-Relevance）
        model_claims = llm_extract_claims(client, question, str(model_out))
        if not model_claims:
            model_claims = [str(model_out).strip()[:3000]]

        # WSI-Relevance：模型 claims vs GT 原文
        relevance = llm_relevance_score(client, model_claims, gt_text)
        # WSI-Precision：GT claims vs 模型输出
        precision = llm_precision_score(client, gt_claims, str(model_out), gt_text)

        r_scores = [float(d.get("score", 0.0)) for d in (relevance or []) if d.get("score") is not None]
        p_scores = [float(d.get("score", 0.0)) for d in (precision or []) if d.get("score") is not None]
        wsi_r_final = (sum(r_scores) / len(r_scores)) if r_scores else None
        wsi_p_final = (sum(p_scores) / len(p_scores)) if p_scores else None

        return ItemResult(
            idx=idx, kept=True, reason="complex_scored", is_mcq=False,
            gt_claims=gt_claims, model_claims=model_claims,
            relevance=relevance, precision=precision,
            wsi_r_final=wsi_r_final, wsi_p_final=wsi_p_final,
            report_metrics=report_metrics,
        )

# ----------------------------
# Aggregations
# ----------------------------

def _avg(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None

def summarize_by_task(results: List[ItemResult], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 新增 “Report Generation”
    cats = ["BCNB", "PathBench", "Morphological Analysis", "Diagnosis", "Treatment Planning", "Report Generation"]
    stat = {
        c: {
            "Open": {"WSI-P": None, "WSI-R": None, "Report-NLU": None, "counts": {"items": 0, "claims": 0, "report_items": 0}},
            "Close": {"Acc": None, "counts": {"items": 0, "correct": 0}},
        } for c in cats
    }

    for i, r in enumerate(results):
        if i >= len(rows):
            break
        if r is None or not r.kept:
            continue
        file_name = str(rows[i].get("file", ""))
        cat = classify_task_by_file(file_name)
        if cat not in stat:
            continue
        if r.is_mcq:
            stat[cat]["Close"]["counts"]["items"] += 1
            if r.mcq_correct:
                stat[cat]["Close"]["counts"]["correct"] += 1
        else:
            # 按样本计数，并累计该样本的 claims 数（用于参考）
            if r.wsi_r_final is not None or r.wsi_p_final is not None:
                stat[cat]["Open"]["counts"]["items"] += 1
                stat[cat]["Open"]["counts"]["claims"] += max(len(r.model_claims or []), len(r.gt_claims or []))
            if r.wsi_r_final is not None:
                stat[cat].setdefault("_R_items", []).append(r.wsi_r_final)
            if r.wsi_p_final is not None:
                stat[cat].setdefault("_P_items", []).append(r.wsi_p_final)
            if cat == "Report Generation" and r.report_metrics:
                stat[cat]["Open"]["counts"]["report_items"] += 1
                pairs = stat[cat].setdefault("_ReportPairs", [])
                ref_text = _get_reference_text(rows[i])
                hyp_text = _get_model_output_text(rows[i])
                pairs.append((ref_text, hyp_text))

    for c in cats:
        rp = stat[c].pop("_R_items", [])
        pp = stat[c].pop("_P_items", [])
        stat[c]["Open"]["WSI-R"] = _avg(rp)  # 平均每样本 final score
        stat[c]["Open"]["WSI-P"] = _avg(pp)
        report_pairs = stat[c].pop("_ReportPairs", [])
        if report_pairs:
            refs, hyps = zip(*report_pairs)
            stat[c]["Open"]["Report-NLU"] = compute_report_metrics_corpus(list(refs), list(hyps))
        else:
            stat[c]["Open"]["Report-NLU"] = None
        ci = stat[c]["Close"]["counts"]["items"]
        cc = stat[c]["Close"]["counts"]["correct"]
        stat[c]["Close"]["Acc"] = (cc / ci) if ci > 0 else None
        # ensure counts dict contains the field even if zero
        stat[c]["Open"]["counts"].setdefault("report_items", 0)

    kept_count = sum(1 for r in results if r and r.kept)
    totals = {
        "total_items": kept_count,
        "by_category_counts": {
            c: {
                "open_items": stat[c]["Open"]["counts"]["items"],
                "open_claims": stat[c]["Open"]["counts"]["claims"],
                "report_items": stat[c]["Open"]["counts"]["report_items"],
                "close_items": stat[c]["Close"]["counts"]["items"],
            } for c in cats
        }
    }
    return {"by_task": stat, "totals": totals}

def summarize_by_file(results: List[ItemResult], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    One record per test json filename:
    - If the file is close-type (choice/TF), output Acc
    - Else (open), output WSI-P / WSI-R
    """
    files: Dict[str, Dict[str, Any]] = {}

    for i, r in enumerate(results):
        if i >= len(rows):
            break
        if r is None or not r.kept:
            continue
        file_name = str(rows[i].get("file", ""))
        if not file_name:
            continue
        task_category = classify_task_by_file(file_name)
        is_report_task = task_category == "Report Generation"
        is_close_file = is_file_close_question(file_name)
        rec = files.setdefault(file_name, {
            "type": "Close" if is_close_file else "Open",
            "Open": {"WSI-P": None, "WSI-R": None, "Report-NLU": None, "counts": {"items": 0, "claims": 0, "report_items": 0}},
            "Close": {"Acc": None, "counts": {"items": 0, "correct": 0}},
        })
        if r.is_mcq:
            rec["Close"]["counts"]["items"] += 1
            if r.mcq_correct:
                rec["Close"]["counts"]["correct"] += 1
        else:
            if r.wsi_r_final is not None or r.wsi_p_final is not None:
                rec["Open"]["counts"]["items"] += 1
                rec["Open"]["counts"]["claims"] += max(len(r.model_claims or []), len(r.gt_claims or []))
            if r.wsi_r_final is not None:
                rec.setdefault("_R_items", []).append(r.wsi_r_final)
            if r.wsi_p_final is not None:
                rec.setdefault("_P_items", []).append(r.wsi_p_final)
            if is_report_task and r.report_metrics:
                rec["Open"]["counts"]["report_items"] += 1
                pairs = rec.setdefault("_ReportPairs", [])
                ref_text = _get_reference_text(rows[i])
                hyp_text = _get_model_output_text(rows[i])
                pairs.append((ref_text, hyp_text))

    # finalize
    out: Dict[str, Any] = {}
    for fname, rec in files.items():
        rp = rec.pop("_R_items", [])
        pp = rec.pop("_P_items", [])
        rec["Open"]["WSI-R"] = _avg(rp)
        rec["Open"]["WSI-P"] = _avg(pp)
        report_pairs = rec.pop("_ReportPairs", [])
        if report_pairs:
            refs, hyps = zip(*report_pairs)
            rec["Open"]["Report-NLU"] = compute_report_metrics_corpus(list(refs), list(hyps))
        else:
            rec["Open"]["Report-NLU"] = None
        ci = rec["Close"]["counts"]["items"]
        cc = rec["Close"]["counts"]["correct"]
        rec["Close"]["Acc"] = (cc / ci) if ci > 0 else None
        rec["Open"]["counts"].setdefault("report_items", 0)
        
        out[fname] = rec
    return out

def load_input_rows(input_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load evaluation rows from either JSONL (default) or CSV files.
    """
    rows: List[Dict[str, Any]] = []
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return rows
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                rows.append(dict(row))
        return rows

    # Default: JSONL input (one JSON object per line)
    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                sys.stderr.write(f"[WARN] Skipping line {i}: invalid JSON.\n")
    return rows

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate WSI LLaVA outputs by task and by file (with TF support).")
    p.add_argument("--input", required=True, help="Path to input JSONL or CSV.")
    p.add_argument("--out-dir", required=True, help="Directory to write outputs.")
    # Provider / API
    p.add_argument("--provider", choices=["openai", "openai-compatible"], default=os.getenv("LLM_PROVIDER", "openai-compatible"))
    p.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3-200b"))
    p.add_argument("--timeout", type=int, default=int(os.getenv("OPENAI_TIMEOUT", "120")))
    p.add_argument("--max-retries", type=int, default=int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    # p.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    p.add_argument("--max-workers", type=int, default=4, help="Concurrency for complex items.")
    p.add_argument("--force-complex", action="store_true", help="Treat all rows as complex (ignore Close fast-path).")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM calls; only filtering + Close.")
    p.add_argument("--report-metric", action="store_true", help="Compute BLEU-1/2/3/4, ROUGE-L, and METEOR for report-generation rows without LLM calls.")
    p.add_argument("--report-metric-llm", action="store_true", help="Report-only mode that computes BLEU-1/2/3/4, ROUGE-L, METEOR and also runs the LLM-based scoring.")
    p.add_argument("--gt-claims-cache-dir", type=str, default=None,
                   help="Directory to cache extracted Ground Truth claims (JSON files). Default: out_dir/gt_claims_cache")
    # Debug limiting
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug).")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N rows (debug).")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for --sample.")
    
    p.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "1.0")))
    p.add_argument("--top-p", type=float, dest="top_p", default=float(os.getenv("OPENAI_TOP_P", "0.8")))
    p.add_argument("--top-k", type=int, dest="top_k", default=int(os.getenv("OPENAI_TOP_K", "20")))
    p.add_argument("--min-p", type=float, dest="min_p", default=float(os.getenv("OPENAI_MIN_P", "0.0")))

    args = p.parse_args()
    if args.report_metric and args.report_metric_llm:
        p.error("Please use either --report-metric or --report-metric-llm (they are mutually exclusive).")

    report_only_mode = args.report_metric or args.report_metric_llm
    metrics_only_mode = args.report_metric

    # Provider defaults
    if args.provider == "openai":
        if not args.api_base:
            args.api_base = "https://api.openai.com/v1"
        if not args.api_key or args.api_key == "EMPTY":
            print("ERROR: --provider openai requires a valid --api-key (or OPENAI_API_KEY).", file=sys.stderr)
            sys.exit(2)
    else:
        if not args.api_base:
            args.api_base = "http://localhost:8000/v1"

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Claims cache directory (default under out_dir)
    if args.gt_claims_cache_dir:
        print('using the caching')
        claims_cache_dir = Path(args.gt_claims_cache_dir)
    else:
        claims_cache_dir = out_dir / "gt_claims_cache"
    claims_cache_dir.mkdir(parents=True, exist_ok=True)

    # Read raw inputs (JSONL or CSV)
    rows = load_input_rows(input_path, args.limit)

    if args.sample is not None:
        random.seed(args.seed)
        if args.sample < len(rows):
            rows = random.sample(rows, args.sample)

    # Reuse existing results.jsonl if possible
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"

    reuse_existing = False
    existing_results: List[ItemResult] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) == len(rows):
            try:
                for ln in lines:
                    d = json.loads(ln)
                    existing_results.append(ItemResult(**d))
                reuse_existing = True
            except Exception:
                reuse_existing = False
    if reuse_existing and report_only_mode:
        needs_report_metrics = False
        for idx, res in enumerate(existing_results):
            file_name = str(rows[idx].get("file", ""))
            if classify_task_by_file(file_name) == "Report Generation" and not res.report_metrics:
                needs_report_metrics = True
                break
        if needs_report_metrics:
            reuse_existing = False
            existing_results = []

    # Evaluate
    results: List[ItemResult]
    if reuse_existing:
        results = existing_results
        print(f"[INFO] Reusing existing results.jsonl with {len(results)} entries; recomputing metrics only.")
    else:
        client = None
        if not args.dry_run and not metrics_only_mode:
            cfg = LLMConfig(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                timeout_s=args.timeout,
                max_retries=args.max_retries,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                provider=args.provider
            )
            client = OpenAICompatClient(cfg)

        tasks_to_run = []
        for idx, row in enumerate(rows):
            # Decide evaluation type based on the file name
            file_name = row.get("file", "")
            is_close = is_file_close_question(file_name)
            
            if args.force_complex:
                is_close = False # Override to treat all as open/complex
                
            tasks_to_run.append({"idx": idx, "row": row, "is_close": is_close})

        results = [None] * len(rows)  # type: ignore
        
        # Separate tasks into simple (Close) and complex (Open)
        simple_tasks = [t for t in tasks_to_run if t["is_close"]]
        complex_tasks = [t for t in tasks_to_run if not t["is_close"]]

        # # Process simple tasks first (no API calls)
        # for task in simple_tasks:
        #     idx = task["idx"]
        #     results[idx] = eval_item(idx, task["row"], client=None, is_close_question=True,
        #                              gt_claims_cache_dir=claims_cache_dir)
            
        # # Process complex tasks with ThreadPoolExecutor
        # if complex_tasks and client:
        #     with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        #         futs = {
        #             ex.submit(eval_item, t["idx"], t["row"], client, is_close_question=False,
        #                       gt_claims_cache_dir=claims_cache_dir): t["idx"]
        #             for t in complex_tasks
        #         }
        #         for fut in as_completed(futs):
        #             idx = futs[fut]
        #             try:
        #                 results[idx] = fut.result()
        #             except Exception as e:
        #                 results[idx] = ItemResult(idx=idx, kept=True, reason=f"complex_error: {e}", is_mcq=False)
        # elif complex_tasks:  # Dry run case for complex tasks
        #     for task in complex_tasks:
        #         idx = task["idx"]
        #         results[idx] = ItemResult(idx=idx, kept=True, reason="complex_but_no_client", is_mcq=False)
         # <<< MODIFIED BLOCK START >>>
        # Wrap the entire processing logic in a tqdm context manager
        with tqdm(total=len(rows), desc="Evaluating items", unit="item") as pbar:
            # Process simple tasks first (no API calls)
            for task in simple_tasks:
                idx = task["idx"]
                results[idx] = eval_item(idx, task["row"], client=None, is_close_question=True,
                                         gt_claims_cache_dir=claims_cache_dir,
                                         report_only_mode=report_only_mode,
                                         metrics_only_mode=metrics_only_mode)
                pbar.update(1)  # Update progress for each simple task
                
            # Process complex tasks with ThreadPoolExecutor
            if complex_tasks and client:
                with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                    futs = {
                        ex.submit(eval_item, t["idx"], t["row"], client, is_close_question=False,
                                  gt_claims_cache_dir=claims_cache_dir,
                                  report_only_mode=report_only_mode,
                                  metrics_only_mode=metrics_only_mode): t["idx"]
                        for t in complex_tasks
                    }
                    for fut in as_completed(futs):
                        idx = futs[fut]
                        try:
                            results[idx] = fut.result()
                        except Exception as e:
                            results[idx] = ItemResult(idx=idx, kept=True, reason=f"complex_error: {e}", is_mcq=False)
                        finally:
                            pbar.update(1) # Update progress for each completed complex task
            
            elif complex_tasks:  # Dry run case for complex tasks
                for task in complex_tasks:
                    idx = task["idx"]
                    results[idx] = eval_item(
                        idx,
                        task["row"],
                        client=None,
                        is_close_question=False,
                        gt_claims_cache_dir=claims_cache_dir,
                        report_only_mode=report_only_mode,
                        metrics_only_mode=metrics_only_mode,
                    )
                    pbar.update(1) # Update progress for each dry-run task
        # <<< MODIFIED BLOCK END >>>

        # Write results.jsonl
        with results_path.open("w", encoding="utf-8") as f:
            for r in results:
                if r is None:
                    r = ItemResult(idx=-1, kept=False, reason="internal_error_unprocessed", is_mcq=False)
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        print(f"[INFO] Results written to: {results_path}")

    # Summaries
    by_task = summarize_by_task(results, rows)
    by_file = summarize_by_file(results, rows)
    summary = {"by_task": by_task["by_task"], "by_file": by_file, "totals": by_task["totals"]}

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[INFO] Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
