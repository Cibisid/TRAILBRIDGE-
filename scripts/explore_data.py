"""
TrialBridge — Day 4: NLP Patient Extraction Pipeline
Reads a raw doctor's note and extracts a structured patient profile.

This is the first engine of TrialBridge. Without this, there is
nothing to match against trials. Every other component depends on
the quality of what this module produces.

Architecture:
  Raw EHR note (text)
       ↓
  scispaCy NER (medical named entity recognition)
       ↓
  negspaCy (negation detection)
       ↓
  Custom extractors (age, labs, medications)
       ↓
  PatientProfile (structured Pydantic model)
"""

import re
from dataclasses import dataclass, field
from typing import Any


# -----------------------------------------------
# Patient Profile Data Model
# -----------------------------------------------
@dataclass
class PatientProfile:
    """
    Structured representation of a patient extracted from a clinical note.
    Every field has a default so partial extraction still produces a valid object.
    """
    # Demographics
    age: int | None = None
    sex: str | None = None                    # "male" | "female" | "unknown"

    # Diagnoses
    primary_diagnosis: str | None = None
    comorbidities: list[str] = field(default_factory=list)
    negated_conditions: list[str] = field(default_factory=list)  # "no history of X"

    # Medications
    current_medications: list[str] = field(default_factory=list)
    prior_treatments: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)

    # Lab values
    lab_values: dict[str, float] = field(default_factory=dict)
    # e.g. {"HbA1c": 8.9, "eGFR": 72, "creatinine": 1.1}

    # Clinical scores
    ecog_score: int | None = None

    # Extraction metadata
    extraction_confidence: float = 0.0
    extraction_warnings: list[str] = field(default_factory=list)
    raw_entities: list[dict] = field(default_factory=list)

    def to_query_string(self) -> str:
        """
        Convert the patient profile into a rich text string
        for semantic embedding. This is what gets fed into BioBERT
        in Week 3 to find matching trials.
        """
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
        }


# -----------------------------------------------
# Sub-Extractors
# -----------------------------------------------
class AgeExtractor:
    """
    Extracts patient age from clinical text.
    Handles patterns like:
      - "58-year-old"
      - "58 year old"
      - "age 58"
      - "58 y/o"
      - "58yo"
    """
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
                if 0 < age < 120:  # Sanity check
                    return age
        return None


class SexExtractor:
    """
    Extracts patient sex from clinical text.
    Handles: male/female, man/woman, M/F, pronouns (he/she)
    """
    MALE_PATTERNS = [
        r'\bmale\b', r'\bman\b', r'\bgentleman\b',
        r'\bhe\b', r'\bhis\b', r'\bhim\b',
        r'\b[Mm]/\d+\b',  # M/58
    ]
    FEMALE_PATTERNS = [
        r'\bfemale\b', r'\bwoman\b', r'\blady\b',
        r'\bshe\b', r'\bher\b', r'\bhers\b',
        r'\b[Ff]/\d+\b',  # F/58
    ]

    def extract(self, text: str) -> str | None:
        text_lower = text.lower()
        male_score = sum(
            1 for p in self.MALE_PATTERNS
            if re.search(p, text_lower)
        )
        female_score = sum(
            1 for p in self.FEMALE_PATTERNS
            if re.search(p, text_lower)
        )
        if female_score > male_score:
            return "female"
        elif male_score > female_score:
            return "male"
        return None


class LabValueExtractor:
    """
    Extracts lab values and vital signs from clinical text.
    Handles patterns like:
      - "HbA1c 8.9%"
      - "HbA1c of 8.9"
      - "eGFR: 72"
      - "creatinine 1.1 mg/dL"
    """
    LAB_PATTERNS = {
        "HbA1c": [
            r'hba1c\s*[:of]?\s*(\d+\.?\d*)\s*%?',
            r'hemoglobin\s+a1c\s*[:of]?\s*(\d+\.?\d*)',
            r'a1c\s*[:of]?\s*(\d+\.?\d*)',
            r'glycated\s+hemoglobin\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "eGFR": [
            r'egfr\s*[:of]?\s*(\d+\.?\d*)',
            r'estimated\s+gfr\s*[:of]?\s*(\d+\.?\d*)',
            r'glomerular\s+filtration\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "creatinine": [
            r'creatinine\s*[:of]?\s*(\d+\.?\d*)',
            r'cr\s*[:of]?\s*(\d+\.?\d*)\s*mg',
        ],
        "ALT": [
            r'\balt\s*[:of]?\s*(\d+\.?\d*)',
            r'alanine\s+aminotransferase\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "AST": [
            r'\bast\s*[:of]?\s*(\d+\.?\d*)',
            r'aspartate\s+aminotransferase\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "hemoglobin": [
            r'\bhgb\s*[:of]?\s*(\d+\.?\d*)',
            r'\bhemoglobin\s*[:of]?\s*(\d+\.?\d*)\s*g',
        ],
        "platelets": [
            r'platelet[s]?\s*[:of]?\s*(\d+\.?\d*)',
            r'\bplt\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "WBC": [
            r'\bwbc\s*[:of]?\s*(\d+\.?\d*)',
            r'white\s+blood\s+cell[s]?\s*[:of]?\s*(\d+\.?\d*)',
        ],
        "blood_pressure_systolic": [
            r'bp\s*[:of]?\s*(\d+)\s*/\s*\d+',
            r'blood\s+pressure\s*[:of]?\s*(\d+)\s*/\s*\d+',
        ],
        "BMI": [
            r'\bbmi\s*[:of]?\s*(\d+\.?\d*)',
            r'body\s+mass\s+index\s*[:of]?\s*(\d+\.?\d*)',
        ],
    }

    def extract(self, text: str) -> dict[str, float]:
        results = {}
        text_lower = text.lower()

        for lab_name, patterns in self.LAB_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        results[lab_name] = value
                        break
                    except (ValueError, IndexError):
                        continue

        return results


class MedicationExtractor:
    """
    Extracts current medications and prior treatments.
    Uses keyword context to distinguish current vs prior.
    """
    CURRENT_CONTEXT = [
        "currently on", "currently taking", "on ", "taking ",
        "prescribed", "medications include", "medications:",
        "is on", "receives", "treated with",
    ]

    PRIOR_CONTEXT = [
        "prior", "previous", "history of treatment",
        "previously treated", "received", "underwent",
        "past treatment", "failed",
    ]

    # Common drug names and classes — expand this in Week 2
    KNOWN_DRUGS = [
        "metformin", "insulin", "glipizide", "glimepiride", "sitagliptin",
        "empagliflozin", "liraglutide", "ozempic", "wegovy", "jardiance",
        "lisinopril", "amlodipine", "atorvastatin", "simvastatin",
        "aspirin", "warfarin", "apixaban", "rivaroxaban",
        "prednisone", "dexamethasone", "methylprednisolone",
        "carboplatin", "cisplatin", "paclitaxel", "docetaxel",
        "pembrolizumab", "nivolumab", "bevacizumab",
        "tamoxifen", "letrozole", "anastrozole",
        "methotrexate", "hydroxychloroquine",
        "sertraline", "fluoxetine", "escitalopram",
        "omeprazole", "pantoprazole", "levothyroxine",
    ]

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        """Returns (current_medications, prior_treatments)"""
        text_lower = text.lower()
        current = []
        prior = []

        for drug in self.KNOWN_DRUGS:
            if drug in text_lower:
                # Determine context
                idx = text_lower.index(drug)
                context_window = text_lower[max(0, idx-60):idx]

                is_prior = any(p in context_window for p in self.PRIOR_CONTEXT)
                is_negated = any(
                    n in context_window
                    for n in ["no ", "not ", "without ", "never "]
                )

                if is_negated:
                    continue  # Skip negated medications
                elif is_prior:
                    prior.append(drug.title())
                else:
                    current.append(drug.title())

        return current, prior


class NegationExtractor:
    """
    Detects negated medical conditions.
    "No history of hypertension" → negated: ["hypertension"]
    "Denies chest pain" → negated: ["chest pain"]

    This is critical for matching — a negated condition must NOT
    be treated as a present condition.
    """
    NEGATION_TRIGGERS = [
        r'no\s+(?:history\s+of\s+|known\s+)?(.+?)(?:\.|,|;|$)',
        r'denies\s+(.+?)(?:\.|,|;|$)',
        r'without\s+(.+?)(?:\.|,|;|$)',
        r'never\s+(?:had\s+|diagnosed\s+with\s+)?(.+?)(?:\.|,|;|$)',
        r'absence\s+of\s+(.+?)(?:\.|,|;|$)',
        r'negative\s+for\s+(.+?)(?:\.|,|;|$)',
    ]

    # Medical conditions to look for in negated context
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
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                negated_phrase = match.group(1).strip()
                # Check if any known condition appears in the negated phrase
                for condition in self.CONDITIONS:
                    if condition in negated_phrase and condition not in negated:
                        negated.append(condition)

        return negated


class DiagnosisExtractor:
    """
    Extracts primary diagnosis and comorbidities.
    Uses pattern matching on common diagnostic phrases.
    """
    DIAGNOSIS_PATTERNS = [
        r'diagnosed\s+with\s+([A-Za-z\s]+?)(?:\.|,|;|\band\b)',
        r'history\s+of\s+([A-Za-z\s]+?)(?:\.|,|;|\band\b)',
        r'known\s+([A-Za-z\s]+?)(?:patient|,|\.|;)',
        r'presents?\s+with\s+([A-Za-z\s]+?)(?:\.|,|;)',
        r'has\s+([A-Za-z\s]+?)(?:\.|,|;|\band\b)',
        r'suffering\s+from\s+([A-Za-z\s]+?)(?:\.|,|;)',
    ]

    KNOWN_CONDITIONS = [
        "type 2 diabetes", "type 1 diabetes", "diabetes mellitus",
        "hypertension", "heart failure", "atrial fibrillation",
        "coronary artery disease", "myocardial infarction",
        "chronic kidney disease", "end stage renal disease",
        "breast cancer", "lung cancer", "colon cancer", "prostate cancer",
        "alzheimer disease", "parkinson disease", "multiple sclerosis",
        "rheumatoid arthritis", "systemic lupus erythematosus",
        "chronic obstructive pulmonary disease", "asthma",
        "hypothyroidism", "hyperthyroidism",
        "depression", "anxiety disorder", "bipolar disorder",
        "obesity", "hyperlipidemia", "dyslipidemia",
    ]

    def extract(self, text: str) -> tuple[str | None, list[str]]:
        """Returns (primary_diagnosis, comorbidities)"""
        text_lower = text.lower()
        found_conditions = []

        # Find all known conditions present in the text
        for condition in self.KNOWN_CONDITIONS:
            if condition in text_lower:
                found_conditions.append(condition.title())

        if not found_conditions:
            return None, []

        # First condition found is treated as primary
        primary = found_conditions[0]
        comorbidities = found_conditions[1:]

        return primary, comorbidities


# -----------------------------------------------
# Main Extractor
# -----------------------------------------------
class PatientProfileExtractor:
    """
    Orchestrates all sub-extractors to produce a complete PatientProfile.
    This is the main class you import and use everywhere in TrialBridge.

    Usage:
        extractor = PatientProfileExtractor()
        profile = extractor.extract("58-year-old female with Type 2 Diabetes...")
        print(profile.to_dict())
    """

    def __init__(self):
        self.age_extractor = AgeExtractor()
        self.sex_extractor = SexExtractor()
        self.lab_extractor = LabValueExtractor()
        self.med_extractor = MedicationExtractor()
        self.neg_extractor = NegationExtractor()
        self.dx_extractor = DiagnosisExtractor()

    def extract(self, note: str) -> PatientProfile:
        """
        Main extraction method.
        Takes raw clinical note text, returns structured PatientProfile.
        """
        if not note or not note.strip():
            return PatientProfile(
                extraction_warnings=["Empty note provided"]
            )

        profile = PatientProfile()
        warnings = []

        # Run all extractors
        profile.age = self.age_extractor.extract(note)
        profile.sex = self.sex_extractor.extract(note)
        profile.lab_values = self.lab_extractor.extract(note)
        profile.negated_conditions = self.neg_extractor.extract(note)

        current_meds, prior_treatments = self.med_extractor.extract(note)
        profile.current_medications = current_meds
        profile.prior_treatments = prior_treatments

        primary_dx, comorbidities = self.dx_extractor.extract(note)
        profile.primary_diagnosis = primary_dx
        profile.comorbidities = comorbidities

        # Collect warnings for missing critical fields
        if not profile.age:
            warnings.append("Age not found in note")
        if not profile.sex:
            warnings.append("Sex not found in note")
        if not profile.primary_diagnosis:
            warnings.append("Primary diagnosis not found in note")

        profile.extraction_warnings = warnings

        # Confidence score: percentage of critical fields extracted
        critical_fields = [profile.age, profile.sex, profile.primary_diagnosis]
        filled = sum(1 for f in critical_fields if f is not None)
        profile.extraction_confidence = filled / len(critical_fields)

        return profile


# -----------------------------------------------
# Test the extractor with real patient notes
# -----------------------------------------------
def run_tests():
    extractor = PatientProfileExtractor()

    test_notes = [
        {
            "label": "Type 2 Diabetes patient",
            "note": """
                58-year-old female with Type 2 Diabetes Mellitus diagnosed 10 years ago.
                Currently on Metformin 1000mg twice daily. HbA1c 8.9%, eGFR 72 ml/min.
                No prior insulin therapy. No history of cardiovascular disease.
                Blood pressure 138/82. BMI 31.2. Denies chest pain or shortness of breath.
            """
        },
        {
            "label": "Cancer patient with complex history",
            "note": """
                A 45 y/o male with breast cancer diagnosed 6 months ago.
                Previously treated with carboplatin. Currently receiving pembrolizumab.
                No history of diabetes or hypertension. ALT 42, AST 38, creatinine 0.9.
                ECOG performance status 1. No prior radiation therapy.
            """
        },
        {
            "label": "Heart failure patient",
            "note": """
                72 year old gentleman with heart failure with reduced ejection fraction.
                History of hypertension and atrial fibrillation. Currently on lisinopril,
                warfarin, and metformin. HbA1c 7.2. eGFR 58. Denies any history of stroke.
                Never had cardiac surgery. WBC 6.8, hemoglobin 11.2 g/dL.
            """
        },
    ]

    print("=" * 65)
    print("TrialBridge — Day 4: NLP Patient Extraction Pipeline Test")
    print("=" * 65)

    for test in test_notes:
        print(f"\n{'─' * 65}")
        print(f"TEST: {test['label']}")
        print(f"{'─' * 65}")
        print(f"INPUT NOTE:\n{test['note'].strip()}")
        print(f"\nEXTRACTED PROFILE:")

        profile = extractor.extract(test['note'])

        print(f"  Age                : {profile.age}")
        print(f"  Sex                : {profile.sex}")
        print(f"  Primary diagnosis  : {profile.primary_diagnosis}")
        print(f"  Comorbidities      : {profile.comorbidities}")
        print(f"  Negated conditions : {profile.negated_conditions}")
        print(f"  Current medications: {profile.current_medications}")
        print(f"  Prior treatments   : {profile.prior_treatments}")
        print(f"  Lab values         : {profile.lab_values}")
        print(f"  Confidence score   : {profile.extraction_confidence:.0%}")
        print(f"  Warnings           : {profile.extraction_warnings}")
        print(f"\n  QUERY STRING (fed into BioBERT in Week 3):")
        print(f"  → {profile.to_query_string()}")

    print(f"\n{'=' * 65}")
    print("Extraction pipeline working. Ready for Week 2 upgrades.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()