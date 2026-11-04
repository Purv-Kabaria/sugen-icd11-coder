"""Common clinical abbreviations and their expansions.

A general primary-care and hospital list, not tuned to any note set. Only entries
whose expansion is a diagnosis or a body system are kept: drug, dosing and
logistics abbreviations add nothing to code retrieval.
"""

from __future__ import annotations

import re
from types import MappingProxyType

ABBREVIATIONS = MappingProxyType(
    {
        "aaa": "abdominal aortic aneurysm",
        "acs": "acute coronary syndrome",
        "adhd": "attention deficit hyperactivity disorder",
        "af": "atrial fibrillation",
        "afib": "atrial fibrillation",
        "aki": "acute kidney injury",
        "all": "acute lymphoblastic leukaemia",
        "aml": "acute myeloid leukaemia",
        "aom": "acute otitis media",
        "ards": "acute respiratory distress syndrome",
        "arf": "acute renal failure",
        "asd": "atrial septal defect",
        "bph": "benign prostatic hyperplasia",
        "bppv": "benign paroxysmal positional vertigo",
        "bv": "bacterial vaginosis",
        "cabg": "coronary artery bypass graft",
        "cad": "coronary artery disease",
        "cap": "community acquired pneumonia",
        "ccf": "congestive cardiac failure",
        "chf": "congestive heart failure",
        "ckd": "chronic kidney disease",
        "cll": "chronic lymphocytic leukaemia",
        "cml": "chronic myeloid leukaemia",
        "copd": "chronic obstructive pulmonary disease",
        "cp": "cerebral palsy",
        "cva": "cerebrovascular accident stroke",
        "dka": "diabetic ketoacidosis",
        "dm": "diabetes mellitus",
        "dm1": "type 1 diabetes mellitus",
        "dm2": "type 2 diabetes mellitus",
        "dvt": "deep vein thrombosis",
        "gad": "generalised anxiety disorder",
        "gbs": "guillain barre syndrome",
        "gdm": "gestational diabetes mellitus",
        "gerd": "gastro oesophageal reflux disease",
        "gord": "gastro oesophageal reflux disease",
        "gi": "gastrointestinal",
        "hcc": "hepatocellular carcinoma",
        "hfref": "heart failure with reduced ejection fraction",
        "hfpef": "heart failure with preserved ejection fraction",
        "htn": "hypertension",
        "hpv": "human papillomavirus",
        "hsv": "herpes simplex",
        "ibd": "inflammatory bowel disease",
        "ibs": "irritable bowel syndrome",
        "ida": "iron deficiency anaemia",
        "ihd": "ischaemic heart disease",
        "itp": "immune thrombocytopenic purpura",
        "lrti": "lower respiratory tract infection",
        "mdd": "major depressive disorder",
        "mi": "myocardial infarction",
        "ms": "multiple sclerosis",
        "nafld": "non alcoholic fatty liver disease",
        "nstemi": "non st elevation myocardial infarction",
        "oa": "osteoarthritis",
        "ocd": "obsessive compulsive disorder",
        "ome": "otitis media with effusion",
        "pcos": "polycystic ovary syndrome",
        "pe": "pulmonary embolism",
        "pet": "pre eclampsia",
        "pih": "pregnancy induced hypertension",
        "pid": "pelvic inflammatory disease",
        "pna": "pneumonia",
        "pprom": "preterm prelabour rupture of membranes",
        "prom": "prelabour rupture of membranes",
        "ptb": "pulmonary tuberculosis",
        "ptsd": "post traumatic stress disorder",
        "pud": "peptic ulcer disease",
        "pvd": "peripheral vascular disease",
        "ra": "rheumatoid arthritis",
        "rhd": "rheumatic heart disease",
        "sam": "severe acute malnutrition",
        "mam": "moderate acute malnutrition",
        "sle": "systemic lupus erythematosus",
        "scd": "sickle cell disease",
        "sti": "sexually transmitted infection",
        "std": "sexually transmitted disease",
        "stemi": "st elevation myocardial infarction",
        "svt": "supraventricular tachycardia",
        "t1dm": "type 1 diabetes mellitus",
        "t2dm": "type 2 diabetes mellitus",
        "tb": "tuberculosis",
        "tia": "transient ischaemic attack",
        "uri": "upper respiratory infection",
        "urti": "upper respiratory tract infection",
        "uti": "urinary tract infection",
        "vsd": "ventricular septal defect",
        "vte": "venous thromboembolism",
        "vt": "ventricular tachycardia",
        "vvc": "vulvovaginal candidiasis",
    }
)

WORD = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")


def expand(text: str) -> str:
    """Append the expansion after each known abbreviation, keeping the original."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        expansion = ABBREVIATIONS.get(word.lower())
        if expansion is None or (word.islower() and len(word) <= 3):
            return word
        return f"{word} {expansion}"

    return WORD.sub(replace, text)
