"""Shared paths and loaders for the Q2 dev pilot."""
import csv
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
Q2 = HERE.parent
VOX = "AudioMemory/voxmembench"
SR = 16000
SEED = "q2-acoustic-residual-2026-10-24"
ACOUSTIC = ("speaker_information", "paralinguistic_information", "environmental_sound")


def splits():
    return {r["question_id"]: r for r in csv.DictReader(open(Q2 / "phase0/splits_8k.csv"))}


def split_ids(split):
    return sorted(q for q, r in splits().items() if r["split"] == split)


def items_csv():
    return {r["question_id"]: r for r in csv.DictReader(open(Q2 / "phase0/out/vox_items_8k.csv"))}


def hkey(*parts):
    """Deterministic pseudo-random key in [0, 1) from SEED and parts."""
    h = hashlib.sha256("|".join([SEED, *map(str, parts)]).encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def load_item(data_dir, qid):
    """Writer-visible item (no evidence roles, no user synthesis text)."""
    return json.load(open(pathlib.Path(data_dir) / "items" / f"{qid}.json"))


def load_eval(data_dir, qid):
    """Analysis-only fields: gold, evidence sessions, session roles, user synthesis text. Never shown to writers."""
    return json.load(open(pathlib.Path(data_dir) / "eval" / f"{qid}.json"))


def gold_text(gold_json, answer_type):
    """Render the gold answer as the string used for teacher-forced log-probabilities."""
    g = json.loads(gold_json)
    if isinstance(g, bool):
        return "Yes" if g else "No"
    if isinstance(g, list):
        return ", ".join(str(x) for x in g)
    return str(g)
