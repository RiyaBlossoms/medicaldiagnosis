"""
Clinical Guidance Engine — maps predictions to recommendations
and appropriate specialist referrals.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class ClinicalGuidance:
    disease: str
    predicted_class: str
    confidence: float
    urgency: str          # "routine" | "urgent" | "emergency"
    recommendations: List[str]
    specialist: str
    disclaimer: str = (
        "This is an AI-assisted finding only. "
        "A qualified clinician must review before any clinical decision."
    )


GUIDANCE_DB = {
    "brain_tumor": {
        "glioma": {
            "urgency": "urgent",
            "specialist": "Neurosurgeon / Neuro-oncologist",
            "recommendations": [
                "Contrast-enhanced MRI with diffusion-weighted imaging",
                "Biopsy for histopathological grading (WHO I–IV)",
                "Multidisciplinary tumor board review",
                "Consider surgical resection, radiotherapy, and temozolomide",
            ],
        },
        "meningioma": {
            "urgency": "routine",
            "specialist": "Neurosurgeon",
            "recommendations": [
                "Serial MRI surveillance every 6–12 months for small lesions",
                "Surgical consult for symptomatic or growing meningiomas",
                "Stereotactic radiosurgery as an alternative for inaccessible lesions",
            ],
        },
        "no_tumor": {
            "urgency": "routine",
            "specialist": "Neurologist (if symptomatic)",
            "recommendations": ["No tumor detected; clinical correlation advised"],
        },
        "pituitary": {
            "urgency": "routine",
            "specialist": "Endocrinologist / Neurosurgeon",
            "recommendations": [
                "Endocrine panel: prolactin, GH, ACTH, TSH, FSH, LH",
                "Visual field testing for macroadenomas",
                "Transsphenoidal surgery for non-functioning macroadenomas",
            ],
        },
    },
    "pneumonia": {
        "normal": {
            "urgency": "routine",
            "specialist": "None required",
            "recommendations": ["Chest X-ray appears clear; correlate with symptoms"],
        },
        "pneumonia": {
            "urgency": "urgent",
            "specialist": "Pulmonologist / Infectious Disease",
            "recommendations": [
                "Sputum culture and sensitivity",
                "CBC, CRP, procalcitonin",
                "Empiric antibiotics per local antibiogram guidelines",
                "SpO2 monitoring; O2 supplementation if < 94%",
                "ICU escalation if sepsis criteria met",
            ],
        },
    },
    "skin_cancer": {
        "mel": {
            "urgency": "urgent",
            "specialist": "Dermatologist / Surgical Oncologist",
            "recommendations": [
                "Urgent dermatoscopy and excisional biopsy",
                "Breslow thickness and sentinel node assessment",
                "Wide-margin excision per staging guidelines",
            ],
        },
        "bcc": {
            "urgency": "routine",
            "specialist": "Dermatologist",
            "recommendations": [
                "Shave or punch biopsy for confirmation",
                "Mohs micrographic surgery for high-risk locations",
            ],
        },
        "nv": {
            "urgency": "routine",
            "specialist": "Dermatologist",
            "recommendations": ["Monitor for ABCDE changes; annual skin check"],
        },
    },
    "eye_disease": {
        "diabetic_retinopathy": {
            "urgency": "urgent",
            "specialist": "Ophthalmologist / Retinal Specialist",
            "recommendations": [
                "Fluorescein angiography and OCT",
                "Optimise glycaemic and BP control",
                "Anti-VEGF injections for DME; laser for proliferative DR",
            ],
        },
        "glaucoma": {
            "urgency": "routine",
            "specialist": "Ophthalmologist",
            "recommendations": [
                "Intraocular pressure measurement",
                "Visual field testing and OCT RNFL",
                "Topical prostaglandin analogs as first-line IOP reduction",
            ],
        },
        "cataract": {
            "urgency": "routine",
            "specialist": "Ophthalmologist",
            "recommendations": [
                "Slit-lamp examination and visual acuity grading",
                "Phacoemulsification when vision significantly impaired",
            ],
        },
        "normal": {
            "urgency": "routine",
            "specialist": "None required",
            "recommendations": ["No significant finding; routine annual eye exam"],
        },
    },
}


def get_guidance(disease: str, predicted_class: str, confidence: float) -> ClinicalGuidance:
    entry = GUIDANCE_DB.get(disease, {}).get(predicted_class, {
        "urgency": "routine",
        "specialist": "General Practitioner",
        "recommendations": ["Consult your healthcare provider"],
    })
    return ClinicalGuidance(
        disease=disease,
        predicted_class=predicted_class,
        confidence=confidence,
        urgency=entry["urgency"],
        recommendations=entry["recommendations"],
        specialist=entry["specialist"],
    )
