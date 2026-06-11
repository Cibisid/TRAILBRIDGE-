"""
TrialBridge — Day 5: NLP Extractor V2
Fixes 2 bugs from Day 4 and adds FastAPI endpoint.

Bug 1 fixed: Negated conditions were being picked as primary diagnosis.
Bug 2 fixed: Duplicate condition detection (Type 2 Diabetes + Diabetes Mellitus).

New: FastAPI endpoint POST /api/v1/parse-patient
"""

import re
from dataclasses import dataclass, field
from typing import Any


# -----------------------------------------------
# Patient Profile (same as Day 4)
# -----------------------------------------------
@dataclass
class PatientProfile:
    age: int | None = None
    sex: str | None = None
    primary_diagnosis: str | None = None
    comorbidities: list[str] = field(default_factory=list)
    negated_conditions: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    prior_treatments: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    lab_values: dict[str, float] = field(default_factory=dict)
    ecog_score: int | None = None
    extraction_confidence: float = 0.0
    extraction_warnings: list[str] = field(default_factory=list)

    def to_query_string(self) -> str:
        parts = []
        if self.age:
            parts.append(f"{self.age} year old patient")
        if self.sex:
            parts.append(self.sex)
        if self.primary_diagnosis:
            parts.append(f"diagnosed with {self.primary_diagnosis}")
        if self.comorbidities:
            parts.append(f"also has {', '.join(self.comorbidities)}")
        if self.current_medications:
            parts.append(f"currently taking {', '.join(self.current_medications)}")
        if self.prior_treatments:
            parts.append(f"prior treatments include {', '.join(self.prior_treatments)}")
        if self.lab_values:
            lab_str = ", ".join(f"{k} {v}" for k, v in self.lab_values.items())
            parts.append(f"lab values: {lab_str}")
        if self.negated_conditions:
            parts.append(f"no history of {', '.join(self.negated_conditions)}")
        return ". ".join(parts) + "." if parts else ""

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "sex": self.sex,
            "primary_diagnosis": self.primary_diagnosis,
            "comorbidities": self.comorbidities,
            "negated_conditions": self.negated_conditions,
            "current_medications": self.current_medications,
            "prior_treatments": self.prior_treatments,
            "allergies": self.allergies,
            "lab_values": self.lab_values,
            "ecog_score": self.ecog_score,
            "extraction_confidence": self.extraction_confidence,
            "extraction_warnings": self.extraction_warnings,
            "query_string": self.to_query_string(),
        }


# -----------------------------------------------
# Bug Fix 1: Negation-Aware Diagnosis Extractor
# -----------------------------------------------
class NegationExtractor:
    """Detects negated conditions first so they can be excluded."""

    NEGATION_TRIGGERS = [
        r'no\s+(?:history\s+of\s+|known\s+)?(.+?)(?:\.|,|;|$)',
        r'denies\s+(.+?)(?:\.|,|;|$)',
        r'without\s+(.+?)(?:\.|,|;|$)',
        r'never\s+(?:had\s+|diagnosed\s+with\s+)?(.+?)(?:\.|,|;|$)',
        r'absence\s+of\s+(.+?)(?:\.|,|;|$)',
        r'negative\s+for\s+(.+?)(?:\.|,|;|$)',
    ]

    CONDITIONS = [
        "hypertension", "diabetes", "cancer", "heart disease",
        "cardiovascular disease", "stroke", "renal disease",
        "liver disease", "hepatitis", "hiv", "tuberculosis",
        "seizures", "epilepsy", "depression", "anxiety",
        "insulin therapy", "chemotherapy", "radiation",
        "surgery", "transplant",
    ]

    def extract(self, text: str) -> list[str]:
        negated = []
        text_lower = text.lower()
        for pattern in self.NEGATION_TRIGGERS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                negated_phrase = match.group(1).strip()
                for condition in self.CONDITIONS:
                    if condition in negated_phrase and condition not in negated:
                        negated.append(condition)
        return negated


class DiagnosisExtractor:
    """
    BUG FIX 1: Now receives negated conditions list and excludes them
    from being picked as primary diagnosis.

    BUG FIX 2: Deduplication — removes conditions that are substrings
    of other conditions (e.g. 'Diabetes' when 'Type 2 Diabetes' exists).
    """

    KNOWN_CONDITIONS = [
        # Most specific first — order matters for primary diagnosis selection
        "type 2 diabetes mellitus",
        "type 2 diabetes",
        "type 1 diabetes",
        "diabetes mellitus",
        "heart failure with reduced ejection fraction",
        "heart failure",
        "atrial fibrillation",
        "coronary artery disease",
        "myocardial infarction",
        "chronic kidney disease",
        "end stage renal disease",
        "breast cancer",
        "lung cancer",
        "colon cancer",
        "prostate cancer",
        "alzheimer disease",
        "parkinson disease",
        "multiple sclerosis",
        "rheumatoid arthritis",
        "systemic lupus erythematosus",
        "chronic obstructive pulmonary disease",
        "hypertension",
        "asthma",
        "hypothyroidism",
        "hyperthyroidism",
        "depression",
        "anxiety disorder",
        "bipolar disorder",
        "obesity",
        "hyperlipidemia",
        "dyslipidemia",
    ]

    def extract(
        self,
        text: str,
        negated_conditions: list[str],
    ) -> tuple[str | None, list[str]]:
        text_lower = text.lower()
        found = []

        for condition in self.KNOWN_CONDITIONS:
            if condition in text_lower:
                # BUG FIX 1: Skip if this condition is negated
                is_negated = any(
                    neg in condition or condition in neg
                    for neg in negated_conditions
                )
                if not is_negated:
                    found.append(condition.title())

        if not found:
            return None, []

        # BUG FIX 2: Remove duplicates — if "Type 2 Diabetes Mellitus"
        # is found, remove "Type 2 Diabetes" and "Diabetes Mellitus"
        # because they refer to the same condition.
        deduplicated = []
        for condition in found:
            is_substring_of_another = any(
                condition.lower() in other.lower()
                for other in found
                if other != condition
            )
            if not is_substring_of_another:
                deduplicated.append(condition)

        primary = deduplicated[0] if deduplicated else None
        comorbidities = deduplicated[1:] if len(deduplicated) > 1 else []

        return primary, comorbidities


# -----------------------------------------------
# Other Extractors (same as Day 4, unchanged)
# -----------------------------------------------
class AgeExtractor:
    PATTERNS = [
        r'(\d+)\s*[-\s]?\s*year[s]?\s*[-\s]?\s*old',
        r'(\d+)\s*y/?o\b',
        r'\bage[d]?\s*:?\s*(\d+)',
        r'(\d+)\s*years?\s*of\s*age',
    ]

    def extract(self, text: str) -> int | None:
        for pattern in self.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 < age < 120:
                    return age
        return None


class SexExtractor:
    MALE_PATTERNS = [r'\bmale\b', r'\bman\b', r'\bgentleman\b', r'\bhe\b', r'\bhis\b']
    FEMALE_PATTERNS = [r'\bfemale\b', r'\bwoman\b', r'\blady\b', r'\bshe\b', r'\bher\b']

    def extract(self, text: str) -> str | None:
        text_lower = text.lower()
        male_score = sum(1 for p in self.MALE_PATTERNS if re.search(p, text_lower))
        female_score = sum(1 for p in self.FEMALE_PATTERNS if re.search(p, text_lower))
        if female_score > male_score:
            return "female"
        elif male_score > female_score:
            return "male"
        return None


class LabValueExtractor:
    LAB_PATTERNS = {
        "HbA1c": [r'hba1c\s*[:of]?\s*(\d+\.?\d*)', r'a1c\s*[:of]?\s*(\d+\.?\d*)'],
        "eGFR": [r'egfr\s*[:of]?\s*(\d+\.?\d*)'],
        "creatinine": [r'creatinine\s*[:of]?\s*(\d+\.?\d*)'],
        "ALT": [r'\balt\s*[:of]?\s*(\d+\.?\d*)'],
        "AST": [r'\bast\s*[:of]?\s*(\d+\.?\d*)'],
        "hemoglobin": [r'\bhgb\s*[:of]?\s*(\d+\.?\d*)', r'\bhemoglobin\s*[:of]?\s*(\d+\.?\d*)\s*g'],
        "platelets": [r'platelet[s]?\s*[:of]?\s*(\d+\.?\d*)'],
        "WBC": [r'\bwbc\s*[:of]?\s*(\d+\.?\d*)'],
        "BMI": [r'\bbmi\s*[:of]?\s*(\d+\.?\d*)'],
        "blood_pressure_systolic": [r'bp\s*[:of]?\s*(\d+)\s*/\s*\d+'],
    }

    def extract(self, text: str) -> dict[str, float]:
        results = {}
        text_lower = text.lower()
        for lab_name, patterns in self.LAB_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        results[lab_name] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        return results


class MedicationExtractor:
    PRIOR_CONTEXT = ["prior", "previous", "previously treated", "received", "underwent"]
    NEGATION_CONTEXT = ["no ", "not ", "without ", "never "]

    KNOWN_DRUGS = [
        "metformin", "insulin", "glipizide", "glimepiride", "sitagliptin",
        "empagliflozin", "liraglutide", "lisinopril", "amlodipine",
        "atorvastatin", "simvastatin", "aspirin", "warfarin", "apixaban",
        "prednisone", "dexamethasone", "carboplatin", "cisplatin",
        "paclitaxel", "docetaxel", "pembrolizumab", "nivolumab",
        "bevacizumab", "tamoxifen", "letrozole", "methotrexate",
        "hydroxychloroquine", "sertraline", "fluoxetine", "omeprazole",
        "levothyroxine",
    ]

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        text_lower = text.lower()
        current, prior = [], []
        for drug in self.KNOWN_DRUGS:
            if drug in text_lower:
                idx = text_lower.index(drug)
                context = text_lower[max(0, idx - 60):idx]
                if any(n in context for n in self.NEGATION_CONTEXT):
                    continue
                elif any(p in context for p in self.PRIOR_CONTEXT):
                    prior.append(drug.title())
                else:
                    current.append(drug.title())
        return current, prior


# -----------------------------------------------
# Main Extractor V2
# -----------------------------------------------
class PatientProfileExtractor:
    """
    V2 — fixes negation bug and duplicate condition bug.
    Drop-in replacement for Day 4 extractor.
    """

    def __init__(self):
        self.age_extractor = AgeExtractor()
        self.sex_extractor = SexExtractor()
        self.lab_extractor = LabValueExtractor()
        self.med_extractor = MedicationExtractor()
        self.neg_extractor = NegationExtractor()
        self.dx_extractor = DiagnosisExtractor()

    def extract(self, note: str) -> PatientProfile:
        if not note or not note.strip():
            return PatientProfile(extraction_warnings=["Empty note provided"])

        profile = PatientProfile()
        warnings = []

        # Run negation FIRST — needed by diagnosis extractor
        profile.negated_conditions = self.neg_extractor.extract(note)

        # Pass negated conditions to diagnosis extractor
        primary_dx, comorbidities = self.dx_extractor.extract(
            note, profile.negated_conditions
        )
        profile.primary_diagnosis = primary_dx
        profile.comorbidities = comorbidities

        # Run remaining extractors
        profile.age = self.age_extractor.extract(note)
        profile.sex = self.sex_extractor.extract(note)
        profile.lab_values = self.lab_extractor.extract(note)
        current_meds, prior_treatments = self.med_extractor.extract(note)
        profile.current_medications = current_meds
        profile.prior_treatments = prior_treatments

        # Warnings for missing fields
        if not profile.age:
            warnings.append("Age not found")
        if not profile.sex:
            warnings.append("Sex not found")
        if not profile.primary_diagnosis:
            warnings.append("Primary diagnosis not found")

        profile.extraction_warnings = warnings

        critical = [profile.age, profile.sex, profile.primary_diagnosis]
        profile.extraction_confidence = sum(
            1 for f in critical if f is not None
        ) / len(critical)

        return profile


# -----------------------------------------------
# FastAPI Endpoint
# -----------------------------------------------
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
_extractor = PatientProfileExtractor()


class ParsePatientRequest(BaseModel):
    note: str


class ParsePatientResponse(BaseModel):
    age: int | None
    sex: str | None
    primary_diagnosis: str | None
    comorbidities: list[str]
    negated_conditions: list[str]
    current_medications: list[str]
    prior_treatments: list[str]
    lab_values: dict[str, float]
    extraction_confidence: float
    extraction_warnings: list[str]
    query_string: str


@router.post(
    "/parse-patient",
    response_model=ParsePatientResponse,
    summary="Parse EHR note into structured patient profile",
    description="""
    Takes a raw clinical note and extracts a structured patient profile.
    The query_string field is ready to be fed into the BioBERT matching engine.
    """,
)
async def parse_patient(request: ParsePatientRequest) -> ParsePatientResponse:
    if not request.note or len(request.note.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Note is too short. Please provide a complete clinical note."
        )

    profile = _extractor.extract(request.note)
    return ParsePatientResponse(**profile.to_dict())


# -----------------------------------------------
# Test V2
# -----------------------------------------------
def run_tests():
    extractor = PatientProfileExtractor()

    test_notes = [
        {
            "label": "Bug Fix 1 — Negated hypertension should NOT be primary diagnosis",
            "note": """
                45 y/o male with breast cancer diagnosed 6 months ago.
                Previously treated with carboplatin. Currently receiving pembrolizumab.
                No history of diabetes or hypertension. ALT 42, AST 38, creatinine 0.9.
            """
        },
        {
            "label": "Bug Fix 2 — No duplicate Type 2 Diabetes + Diabetes Mellitus",
            "note": """
                58-year-old female with Type 2 Diabetes Mellitus.
                Currently on Metformin. HbA1c 8.9%, eGFR 72.
                No prior insulin therapy. No history of cardiovascular disease.
            """
        },
        {
            "label": "Full extraction test — Heart failure patient",
            "note": """
                72 year old gentleman with heart failure with reduced ejection fraction.
                History of hypertension and atrial fibrillation.
                Currently on lisinopril, warfarin, and metformin.
                HbA1c 7.2. eGFR 58. Denies any history of stroke.
                WBC 6.8, hemoglobin 11.2 g/dL.
            """
        },
    ]

    print("=" * 65)
    print("TrialBridge — Day 5: Extractor V2 Bug Fix Tests")
    print("=" * 65)

    all_passed = True

    for test in test_notes:
        print(f"\n{'─' * 65}")
        print(f"TEST: {test['label']}")
        print(f"{'─' * 65}")

        profile = extractor.extract(test['note'])

        print(f"  Primary diagnosis  : {profile.primary_diagnosis}")
        print(f"  Comorbidities      : {profile.comorbidities}")
        print(f"  Negated conditions : {profile.negated_conditions}")
        print(f"  Current medications: {profile.current_medications}")
        print(f"  Prior treatments   : {profile.prior_treatments}")
        print(f"  Lab values         : {profile.lab_values}")
        print(f"  Confidence         : {profile.extraction_confidence:.0%}")
        print(f"  Warnings           : {profile.extraction_warnings}")
        print(f"  Query string       : {profile.to_query_string()[:100]}...")

        # Assertions
        if "Bug Fix 1" in test['label']:
            assert profile.primary_diagnosis == "Breast Cancer", \
                f"FAIL: Expected 'Breast Cancer', got '{profile.primary_diagnosis}'"
            assert "hypertension" in profile.negated_conditions, \
                "FAIL: Hypertension should be negated"
            print("  PASS — Negated condition not picked as primary diagnosis")

        if "Bug Fix 2" in test['label']:
            conditions = [profile.primary_diagnosis] + profile.comorbidities
            diabetes_variants = [
                c for c in conditions
                if c and "diabetes" in c.lower()
            ]
            assert len(diabetes_variants) == 1, \
                f"FAIL: Expected 1 diabetes condition, got {diabetes_variants}"
            print("  PASS — No duplicate diabetes conditions")

    print(f"\n{'=' * 65}")
    print("All bug fixes verified. Extractor V2 ready.")
    print("FastAPI endpoint registered at POST /api/v1/parse-patient")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()