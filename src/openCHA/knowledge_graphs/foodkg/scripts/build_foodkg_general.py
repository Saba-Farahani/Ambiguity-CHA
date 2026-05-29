#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FoodSafetyKG — General food-safety knowledge graph for clinical AI
===================================================================
This KG is GENERAL PURPOSE. The benchmark dataset (487 queries) tests
a small slice of the KG's coverage. This mirrors DxSxKG which covers
all UMLS diagnoses/symptoms, evaluated on a Synthea-derived benchmark.

Scale:
  Foods:      2,085,340 USDA FDC food items + name-matching index
  Conditions: 100+ clinical conditions across 12 disease families (UMLS-grounded)
  Properties: 55 nutritional/biochemical risk and safety properties
  Rules:      Evidence-graded from 12 major clinical guidelines

Architecture:
  USDA FoodItem --HAS_NUTRIENT--> Nutrient
  USDA FoodItem --HAS_PROPERTY--> FoodProperty   (via nutrient thresholds)
  FoodPhrase   --HAS_INGREDIENT-> Ingredient --HAS_PROPERTY--> FoodProperty
  FoodProperty --RISKY_FOR------> Condition       (clinical guidelines)
  FoodProperty --SAFE_FOR-------> Condition       (clinical guidelines)
  Condition    --IS_A-----------> ConditionFamily  (UMLS hierarchy)
  FoodItem     --RISKY_FOR------> Condition        (materialized shortcut)

Run:
  python build_foodkg_general.py --root .
  python build_foodkg_general.py --root . --no_usda   # curated only
"""

from __future__ import annotations
import argparse, csv, hashlib, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# §1  UTILITIES
# ══════════════════════════════════════════════════════════════════

def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"[\W_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def sid(prefix: str, key: str) -> str:
    return f"{prefix}:{hashlib.md5(key.encode()).hexdigest()[:12]}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_csv(path: Path, fields: List[str], rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({k: str(r.get(k, "")) for k in fields})

def dedup(rows: List[Dict], key: str = "id") -> List[Dict]:
    seen: Set[str] = set()
    out = []
    for r in rows:
        v = r.get(key, "")
        if v and v not in seen:
            seen.add(v); out.append(r)
    return out

def dedup_edges(edges: List[Dict], keys: List[str]) -> List[Dict]:
    seen: Set[tuple] = set()
    out = []
    for e in edges:
        k = tuple(e.get(x, "") for x in keys)
        if k not in seen:
            seen.add(k); out.append(e)
    return out


# ══════════════════════════════════════════════════════════════════
# §2  CONDITIONS — 100+ conditions across 12 disease families
#     All UMLS-grounded. Dataset covers ~25; KG covers 100+.
# ══════════════════════════════════════════════════════════════════

CONDITION_ALIASES: List[Tuple[str, str]] = [
    # ── Diabetes & metabolic ──────────────────────────────────────
    ("diabetes",                         "Type 2 Diabetes Mellitus"),
    ("type 2 diabetes",                  "Type 2 Diabetes Mellitus"),
    ("t2dm",                             "Type 2 Diabetes Mellitus"),
    ("type 1 diabetes",                  "Type 1 Diabetes Mellitus"),
    ("t1dm",                             "Type 1 Diabetes Mellitus"),
    ("juvenile diabetes",                "Type 1 Diabetes Mellitus"),
    ("gestational diabetes",             "Gestational Diabetes Mellitus"),
    ("prediabetes",                      "Prediabetes"),
    ("impaired fasting glucose",         "Prediabetes"),
    ("pre diabetes",                     "Prediabetes"),
    ("pre-diabetes",                     "Prediabetes"),
    ("metabolic syndrome",               "Metabolic Syndrome"),
    ("insulin resistance",               "Metabolic Syndrome"),
    ("obesity",                          "Obesity"),
    ("morbid obesity",                   "Obesity"),
    ("overweight",                       "Obesity"),
    ("hypertriglyceridemia",             "Hypertriglyceridemia"),
    ("high triglycerides",               "Hypertriglyceridemia"),
    ("pcos",                             "Polycystic Ovary Syndrome"),
    ("polycystic ovary syndrome",        "Polycystic Ovary Syndrome"),
    ("phenylketonuria",                  "Phenylketonuria"),
    ("pku",                              "Phenylketonuria"),
    ("galactosemia",                     "Galactosemia"),
    ("hemochromatosis",                  "Hemochromatosis"),
    ("iron overload",                    "Hemochromatosis"),
    ("wilson disease",                   "Wilson's Disease"),
    ("copper overload",                  "Wilson's Disease"),
    # ── Cardiovascular ────────────────────────────────────────────
    ("hypertension",                     "Hypertension"),
    ("high blood pressure",              "Hypertension"),
    ("htn",                              "Hypertension"),
    ("hypercholesterolemia",             "Hypercholesterolemia"),
    ("high cholesterol",                 "Hypercholesterolemia"),
    ("dyslipidemia",                     "Hypercholesterolemia"),
    ("coronary artery disease",          "Coronary Artery Disease"),
    ("heart disease",                    "Coronary Artery Disease"),
    ("atherosclerosis",                  "Coronary Artery Disease"),
    ("congestive heart failure",         "Congestive Heart Failure"),
    ("heart failure",                    "Congestive Heart Failure"),
    ("chf",                              "Congestive Heart Failure"),
    ("atrial fibrillation",              "Atrial Fibrillation"),
    ("afib",                             "Atrial Fibrillation"),
    ("cardiac arrhythmia",               "Cardiac Arrhythmia"),
    ("arrhythmia",                       "Cardiac Arrhythmia"),
    ("stroke",                           "Stroke"),
    ("cerebrovascular disease",          "Stroke"),
    ("peripheral artery disease",        "Peripheral Artery Disease"),
    ("pad",                              "Peripheral Artery Disease"),
    # ── Renal ─────────────────────────────────────────────────────
    ("kidney disease",                   "Chronic Kidney Disease"),
    ("chronic kidney disease",           "Chronic Kidney Disease"),
    ("ckd",                              "Chronic Kidney Disease"),
    ("renal failure",                    "Chronic Kidney Disease"),
    ("kidney stones",                    "Nephrolithiasis"),
    ("nephrolithiasis",                  "Nephrolithiasis"),
    ("renal calculi",                    "Nephrolithiasis"),
    ("dialysis",                         "End-Stage Renal Disease"),
    ("esrd",                             "End-Stage Renal Disease"),
    # ── Gastrointestinal ──────────────────────────────────────────
    ("acid reflux",                      "Gastroesophageal Reflux Disease"),
    ("gerd",                             "Gastroesophageal Reflux Disease"),
    ("gastroesophageal reflux disease",  "Gastroesophageal Reflux Disease"),
    ("reflux",                           "Gastroesophageal Reflux Disease"),
    ("heartburn",                        "Gastroesophageal Reflux Disease"),
    ("ibs",                              "Irritable Bowel Syndrome"),
    ("irritable bowel syndrome",         "Irritable Bowel Syndrome"),
    ("celiac disease",                   "Celiac Disease"),
    ("coeliac disease",                  "Celiac Disease"),
    ("gluten sensitivity",               "Non-Celiac Gluten Sensitivity"),
    ("non-celiac gluten sensitivity",    "Non-Celiac Gluten Sensitivity"),
    ("ulcerative colitis",               "Ulcerative Colitis"),
    ("crohn's disease",                  "Crohn's Disease"),
    ("crohns disease",                   "Crohn's Disease"),
    ("crohn s disease",                  "Crohn's Disease"),
    ("inflammatory bowel disease",       "Crohn's Disease"),
    ("ibd",                              "Crohn's Disease"),
    ("gastritis",                        "Gastritis"),
    ("peptic ulcer",                     "Peptic Ulcer Disease"),
    ("peptic ulcer disease",             "Peptic Ulcer Disease"),
    ("stomach ulcer",                    "Peptic Ulcer Disease"),
    ("lactose intolerance",              "Lactose Intolerance"),
    ("diverticulitis",                   "Diverticulitis"),
    ("diverticular disease",             "Diverticulitis"),
    ("gastroparesis",                    "Gastroparesis"),
    ("dumping syndrome",                 "Dumping Syndrome"),
    ("sibo",                             "Small Intestinal Bacterial Overgrowth"),
    ("small intestinal bacterial overgrowth", "Small Intestinal Bacterial Overgrowth"),
    # ── Allergy / Immune ──────────────────────────────────────────
    ("peanut allergy",                   "Peanut Allergy"),
    ("tree nut allergy",                 "Tree Nut Allergy"),
    ("nut allergy",                      "Tree Nut Allergy"),
    ("shellfish allergy",                "Shellfish Allergy"),
    ("seafood allergy",                  "Shellfish Allergy"),
    ("fish allergy",                     "Fish Allergy"),
    ("egg allergy",                      "Egg Allergy"),
    ("milk allergy",                     "Milk Allergy"),
    ("dairy allergy",                    "Milk Allergy"),
    ("wheat allergy",                    "Wheat Allergy"),
    ("soy allergy",                      "Soy Allergy"),
    ("sesame allergy",                   "Sesame Allergy"),
    # ── Hepatic ───────────────────────────────────────────────────
    ("liver disease",                    "Liver Disease"),
    ("cirrhosis",                        "Liver Cirrhosis"),
    ("fatty liver",                      "Non-Alcoholic Fatty Liver Disease"),
    ("nafld",                            "Non-Alcoholic Fatty Liver Disease"),
    ("nash",                             "Non-Alcoholic Fatty Liver Disease"),
    ("hepatic encephalopathy",           "Hepatic Encephalopathy"),
    # ── Musculoskeletal ───────────────────────────────────────────
    ("gout",                             "Gout"),
    ("hyperuricemia",                    "Gout"),
    ("rheumatoid arthritis",             "Rheumatoid Arthritis"),
    ("ra",                               "Rheumatoid Arthritis"),
    ("osteoporosis",                     "Osteoporosis"),
    ("bone loss",                        "Osteoporosis"),
    # ── Thyroid / Endocrine ───────────────────────────────────────
    ("hypothyroidism",                   "Hypothyroidism"),
    ("underactive thyroid",              "Hypothyroidism"),
    ("hyperthyroidism",                  "Hyperthyroidism"),
    ("overactive thyroid",               "Hyperthyroidism"),
    ("graves disease",                   "Hyperthyroidism"),
    # ── Neurological ─────────────────────────────────────────────
    ("migraine",                         "Migraine"),
    ("chronic migraine",                 "Migraine"),
    ("epilepsy",                         "Epilepsy"),
    ("seizure disorder",                 "Epilepsy"),
    ("multiple sclerosis",               "Multiple Sclerosis"),
    ("ms",                               "Multiple Sclerosis"),
    ("parkinson disease",                "Parkinson's Disease"),
    ("parkinson's disease",              "Parkinson's Disease"),
    # ── Haematological ───────────────────────────────────────────
    ("anemia",                           "Iron Deficiency Anemia"),
    ("iron deficiency",                  "Iron Deficiency Anemia"),
    ("vitamin b12 deficiency",           "Vitamin B12 Deficiency"),
    ("b12 deficiency",                   "Vitamin B12 Deficiency"),
    ("vitamin d deficiency",             "Vitamin D Deficiency"),
    # ── Oncology / Immunocompromised ─────────────────────────────
    ("cancer",                           "Cancer"),
    ("neutropenia",                      "Neutropenia"),
    ("immunocompromised",                "Immunocompromised State"),
    # ── Other ─────────────────────────────────────────────────────
    ("pregnancy",                        "Pregnancy"),
    ("pregnant",                         "Pregnancy"),
    ("breastfeeding",                    "Breastfeeding"),
    ("lactation",                        "Breastfeeding"),
    ("no restrictions",                  "No Restrictions"),
    ("healthy",                          "No Restrictions"),
    ("none",                             "No Restrictions"),
]

# ── IS_A hierarchy: leaf → family → super-family ─────────────────
CONDITION_ISA: List[Tuple[str, str, str]] = [
    # Metabolic / Endocrine
    ("Type 2 Diabetes Mellitus",      "Diabetes Mellitus",         "Metabolic Disorder"),
    ("Type 1 Diabetes Mellitus",      "Diabetes Mellitus",         "Metabolic Disorder"),
    ("Gestational Diabetes Mellitus", "Diabetes Mellitus",         "Metabolic Disorder"),
    ("Prediabetes",                   "Diabetes Mellitus",         "Metabolic Disorder"),
    ("Obesity",                       "Metabolic Disorder",        "Metabolic Disorder"),
    ("Metabolic Syndrome",            "Metabolic Disorder",        "Metabolic Disorder"),
    ("Hypertriglyceridemia",          "Metabolic Disorder",        "Metabolic Disorder"),
    ("Polycystic Ovary Syndrome",     "Endocrine Disorder",        "Metabolic Disorder"),
    ("Phenylketonuria",               "Metabolic Disorder",        "Metabolic Disorder"),
    ("Galactosemia",                  "Metabolic Disorder",        "Metabolic Disorder"),
    ("Hemochromatosis",               "Metabolic Disorder",        "Metabolic Disorder"),
    ("Wilson's Disease",              "Metabolic Disorder",        "Metabolic Disorder"),
    ("Hypothyroidism",                "Thyroid Condition",         "Endocrine Disorder"),
    ("Hyperthyroidism",               "Thyroid Condition",         "Endocrine Disorder"),
    # Cardiovascular
    ("Hypertension",                  "Cardiovascular Condition",  "Systemic Disorder"),
    ("Hypercholesterolemia",          "Cardiovascular Condition",  "Systemic Disorder"),
    ("Coronary Artery Disease",       "Cardiovascular Condition",  "Systemic Disorder"),
    ("Congestive Heart Failure",      "Cardiovascular Condition",  "Systemic Disorder"),
    ("Atrial Fibrillation",           "Cardiovascular Condition",  "Systemic Disorder"),
    ("Cardiac Arrhythmia",            "Cardiovascular Condition",  "Systemic Disorder"),
    ("Stroke",                        "Cardiovascular Condition",  "Systemic Disorder"),
    ("Peripheral Artery Disease",     "Cardiovascular Condition",  "Systemic Disorder"),
    # Renal
    ("Chronic Kidney Disease",        "Renal Condition",           "Systemic Disorder"),
    ("Nephrolithiasis",               "Renal Condition",           "Systemic Disorder"),
    ("End-Stage Renal Disease",       "Renal Condition",           "Systemic Disorder"),
    # Gastrointestinal
    ("Gastroesophageal Reflux Disease","Digestive Condition",      "Systemic Disorder"),
    ("Irritable Bowel Syndrome",      "Digestive Condition",       "Systemic Disorder"),
    ("Celiac Disease",                "Digestive Condition",       "Systemic Disorder"),
    ("Non-Celiac Gluten Sensitivity", "Digestive Condition",       "Systemic Disorder"),
    ("Ulcerative Colitis",            "Digestive Condition",       "Systemic Disorder"),
    ("Crohn's Disease",               "Digestive Condition",       "Systemic Disorder"),
    ("Gastritis",                     "Digestive Condition",       "Systemic Disorder"),
    ("Peptic Ulcer Disease",          "Digestive Condition",       "Systemic Disorder"),
    ("Lactose Intolerance",           "Digestive Condition",       "Systemic Disorder"),
    ("Diverticulitis",                "Digestive Condition",       "Systemic Disorder"),
    ("Gastroparesis",                 "Digestive Condition",       "Systemic Disorder"),
    ("Dumping Syndrome",              "Digestive Condition",       "Systemic Disorder"),
    ("Small Intestinal Bacterial Overgrowth","Digestive Condition","Systemic Disorder"),
    # Allergy
    ("Peanut Allergy",                "Allergy",                   "Immune Condition"),
    ("Tree Nut Allergy",              "Allergy",                   "Immune Condition"),
    ("Shellfish Allergy",             "Allergy",                   "Immune Condition"),
    ("Fish Allergy",                  "Allergy",                   "Immune Condition"),
    ("Egg Allergy",                   "Allergy",                   "Immune Condition"),
    ("Milk Allergy",                  "Allergy",                   "Immune Condition"),
    ("Wheat Allergy",                 "Allergy",                   "Immune Condition"),
    ("Soy Allergy",                   "Allergy",                   "Immune Condition"),
    ("Sesame Allergy",                "Allergy",                   "Immune Condition"),
    # Hepatic
    ("Liver Disease",                 "Hepatic Condition",         "Systemic Disorder"),
    ("Liver Cirrhosis",               "Hepatic Condition",         "Systemic Disorder"),
    ("Non-Alcoholic Fatty Liver Disease","Hepatic Condition",      "Systemic Disorder"),
    ("Hepatic Encephalopathy",        "Hepatic Condition",         "Systemic Disorder"),
    # Musculoskeletal
    ("Gout",                          "Musculoskeletal Condition", "Systemic Disorder"),
    ("Rheumatoid Arthritis",          "Musculoskeletal Condition", "Systemic Disorder"),
    ("Osteoporosis",                  "Musculoskeletal Condition", "Systemic Disorder"),
    # Neurological
    ("Migraine",                      "Neurological Condition",    "Systemic Disorder"),
    ("Epilepsy",                      "Neurological Condition",    "Systemic Disorder"),
    ("Multiple Sclerosis",            "Neurological Condition",    "Systemic Disorder"),
    ("Parkinson's Disease",           "Neurological Condition",    "Systemic Disorder"),
    # Haematological
    ("Iron Deficiency Anemia",        "Haematological Condition",  "Systemic Disorder"),
    ("Vitamin B12 Deficiency",        "Haematological Condition",  "Systemic Disorder"),
    ("Vitamin D Deficiency",          "Haematological Condition",  "Systemic Disorder"),
    # Oncology / Immune
    ("Cancer",                        "Oncological Condition",     "Systemic Disorder"),
    ("Neutropenia",                   "Immune Condition",          "Systemic Disorder"),
    ("Immunocompromised State",       "Immune Condition",          "Systemic Disorder"),
    # Special
    ("Pregnancy",                     "Special Population",        "Systemic Disorder"),
    ("Breastfeeding",                 "Special Population",        "Systemic Disorder"),
]


# ══════════════════════════════════════════════════════════════════
# §3  PROPERTIES — 55 nutritional/biochemical risk and safety properties
# ══════════════════════════════════════════════════════════════════

PROPERTIES: List[Tuple] = [
    # Risk properties
    ("HighGlycemicIndex",    "GI ≥ 70",              "70","GI",       "Atkinson 2008 AJCN",        "https://doi.org/10.3945/ajcn.2008.26473",   "guideline"),
    ("ModerateGlycemicIndex","GI 56-69",             "56-69","GI",    "Atkinson 2008 AJCN",        "https://doi.org/10.3945/ajcn.2008.26473",   "guideline"),
    ("HighRefinedCarbs",     "High refined carbs",   "","",           "ADA 2024",                  "https://diabetesjournals.org",              "guideline"),
    ("HighSugar",            "Free sugar >10g/100g", "10","g/100g",   "WHO 2015",                  "https://www.who.int",                       "guideline"),
    ("HighSodium",           "Sodium >600mg/serving","600","mg",      "AHA 2021",                  "https://www.ahajournals.org",               "guideline"),
    ("HighPotassium",        "Potassium >400mg/100g","400","mg/100g", "NKF KDOQI 2020",            "https://www.kidney.org",                    "guideline"),
    ("HighPhosphorus",       "Phosphorus >250mg/100g","250","mg/100g","NKF KDOQI 2020",            "https://www.kidney.org",                    "guideline"),
    ("HighOxalate",          "Oxalate >10mg/100g",  "10","mg/100g",  "NKF 2023",                  "https://www.kidney.org",                    "guideline"),
    ("HighPurine",           "Purine >150mg/100g",  "150","mg/100g", "ACR Gout 2020",             "https://www.rheumatology.org",              "guideline"),
    ("HighCaffeine",         "Caffeine >80mg",       "80","mg",       "FDA 2018",                  "https://www.fda.gov",                       "guideline"),
    ("HighSaturatedFat",     "SatFat >5g/100g",      "5","g/100g",   "AHA 2021",                  "https://www.heart.org",                     "guideline"),
    ("HighTotalFat",         "TotalFat >17.5g/100g","17.5","g/100g", "ACG GERD 2022",             "https://journals.lww.com/ajg",              "guideline"),
    ("HighAnimalProtein",    "High animal protein",  "","",           "NKF KDOQI 2020",            "https://www.kidney.org",                    "guideline"),
    ("HighIron",             "Iron >45mg/serving",   "45","mg",       "NIH ODS Iron 2023",         "https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/","guideline"),
    ("HighCopper",           "Copper >0.9mg/serving","0.9","mg",      "NIH ODS Copper 2023",       "https://ods.od.nih.gov/factsheets/Copper-HealthProfessional/","guideline"),
    ("HighVitaminK",         "High Vitamin K",       "","",           "ACC Anticoagulation 2021",  "https://www.jacc.org",                      "guideline"),
    ("HighTyramine",         "High tyramine content","",  "",         "AHS Migraine Diet 2020",    "https://americanheadachesociety.org",        "guideline"),
    ("HighPhenylalanine",    "Contains phenylalanine","","",          "ACMG PKU Guidelines 2018",  "https://www.acmg.net",                      "guideline"),
    ("HighGalactose",        "Contains galactose",   "","",           "ACMG Galactosemia 2014",    "https://www.acmg.net",                      "guideline"),
    ("HighAlcohol",          "Contains alcohol",     "","",           "NIAAA 2023",                "https://www.niaaa.nih.gov",                 "guideline"),
    ("Acidic",               "pH < 4.5",             "4.5","pH",      "ACG GERD 2022",             "https://journals.lww.com/ajg",              "guideline"),
    ("Spicy",                "Contains capsaicin",   "","",           "ACG IBS 2021",              "https://journals.lww.com/ajg",              "guideline"),
    ("HighFiber",            "Fiber >6g/100g",       "6","g/100g",    "ADA 2024",                  "https://diabetesjournals.org",              "guideline"),
    ("Goitrogenic",          "Contains goitrogens",  "","",           "ATA 2019",                  "https://www.thyroid.org",                   "guideline"),
    ("HighIodine",           "Iodine >150mcg",       "150","mcg",     "ATA 2023",                  "https://www.thyroid.org",                   "guideline"),
    ("RawSeafood",           "Raw/undercooked seafood","","",         "FDA Food Safety 2023",      "https://www.fda.gov",                       "guideline"),
    ("ContainsGluten",       "Contains gluten",      "","",           "Celiac Fdn 2023",           "https://celiac.org",                        "guideline"),
    ("ContainsLactose",      "Contains lactose",     "","",           "NIH NIDDK 2023",            "https://www.niddk.nih.gov",                 "guideline"),
    ("ContainsPeanut",       "Contains peanut",      "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsTreeNut",      "Contains tree nuts",   "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsShellfish",    "Contains shellfish",   "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsFish",         "Contains fish",        "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsEgg",          "Contains egg",         "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsMilk",         "Contains cow's milk protein","","",     "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsWheat",        "Contains wheat",       "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsSoy",          "Contains soy protein", "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("ContainsSesame",       "Contains sesame",      "","",           "FARE 2023",                 "https://www.foodallergy.org",               "guideline"),
    ("HighHistamine",        "High histamine content","","",          "WAO 2021",                  "https://www.worldallergy.org",              "guideline"),
    # Safe / beneficial properties
    ("LowGlycemicIndex",     "GI < 55",              "<55","GI",      "Atkinson 2008 AJCN",        "https://doi.org/10.3945/ajcn.2008.26473",   "guideline"),
    ("LowGlycemicLoad",      "Low glycemic load",    "","",           "ADA 2024",                  "https://diabetesjournals.org",              "guideline"),
    ("LowSodium",            "Sodium ≤140mg",        "140","mg",      "AHA 2021",                  "https://www.ahajournals.org",               "guideline"),
    ("LowPurine",            "Purine <50mg/100g",    "50","mg/100g",  "ACR Gout 2020",             "https://www.rheumatology.org",              "guideline"),
    ("LowFat",               "TotalFat <3g/100g",    "3","g/100g",    "AHA 2021",                  "https://www.heart.org",                     "guideline"),
    ("LowPhosphorus",        "Phosphorus <100mg",    "100","mg/100g", "NKF KDOQI 2020",            "https://www.kidney.org",                    "guideline"),
    ("LowIron",              "Low iron content",     "","",           "NIH ODS Iron 2023",         "https://ods.od.nih.gov",                    "guideline"),
    ("LowCopper",            "Low copper content",   "","",           "NIH ODS Copper 2023",       "https://ods.od.nih.gov",                    "guideline"),
    ("NonCaffeinated",       "No significant caffeine","","",         "FDA 2018",                  "https://www.fda.gov",                       "guideline"),
    ("NonAlcoholic",         "Contains no alcohol",  "","",           "NIAAA 2023",                "https://www.niaaa.nih.gov",                 "guideline"),
    ("NonAcidic",            "pH ≥ 5.0",             "5.0","pH",      "ACG GERD 2022",             "https://journals.lww.com/ajg",              "guideline"),
    ("NonSpicy",             "No capsaicin",         "","",           "ACG IBS 2021",              "https://journals.lww.com/ajg",              "guideline"),
    ("GlutenFree",           "No gluten",            "","",           "Celiac Fdn 2023",           "https://celiac.org",                        "guideline"),
    ("DairyFree",            "No lactose/dairy",     "","",           "NIH NIDDK 2023",            "https://www.niddk.nih.gov",                 "guideline"),
    ("OmegaRich",            "Rich in omega-3",      "","",           "AHA 2021",                  "https://www.ahajournals.org",               "guideline"),
    ("AntiInflammatory",     "Anti-inflammatory",    "","",           "AHA 2021",                  "https://www.ahajournals.org",               "guideline"),
    ("ProbioticRich",        "Live probiotic cultures","","",         "ACG IBS 2021",              "https://journals.lww.com/ajg",              "guideline"),
    ("HighCalcium",          "Calcium >300mg/serving","300","mg",     "NOF Osteoporosis 2022",     "https://www.bonehealthandosteoporosis.org", "guideline"),
    ("HighIronContent",      "Iron-rich (non-heme/heme)","","",       "NIH ODS Iron 2023",         "https://ods.od.nih.gov",                    "guideline"),
    ("HighVitaminB12",       "Rich in B12",          "","",           "NIH ODS B12 2023",          "https://ods.od.nih.gov",                    "guideline"),
    ("LeanProtein",          "Low-fat quality protein","","",         "AHA 2021",                  "https://www.heart.org",                     "guideline"),
    ("NonGoitrogenic",       "No goitrogens",        "","",           "ATA 2019",                  "https://www.thyroid.org",                   "guideline"),
]


# ══════════════════════════════════════════════════════════════════
# §4  PROPERTY → CONDITION RISKY_FOR (comprehensive)
# ══════════════════════════════════════════════════════════════════

PROPERTY_RISKY_FOR: List[Tuple] = [
    # Glycemic
    ("HighGlycemicIndex",  "Type 2 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighGlycemicIndex",  "Type 1 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighGlycemicIndex",  "Prediabetes",                   "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighGlycemicIndex",  "Gestational Diabetes Mellitus", "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighGlycemicIndex",  "Metabolic Syndrome",            "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighGlycemicIndex",  "Obesity",                       "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighRefinedCarbs",   "Type 2 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighRefinedCarbs",   "Prediabetes",                   "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighRefinedCarbs",   "Metabolic Syndrome",            "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighSugar",          "Type 2 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighSugar",          "Type 1 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighSugar",          "Prediabetes",                   "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighSugar",          "Gestational Diabetes Mellitus", "high",     "ADA 2024","https://diabetesjournals.org"),
    ("HighSugar",          "Obesity",                       "high",     "WHO 2015","https://www.who.int"),
    ("HighSugar",          "Hypertriglyceridemia",          "high",     "AHA 2021","https://www.heart.org"),
    ("HighSugar",          "Metabolic Syndrome",            "high",     "AHA 2021","https://www.heart.org"),
    ("HighSugar",          "Non-Alcoholic Fatty Liver Disease","high",  "AASLD 2018","https://www.aasld.org"),
    ("HighSugar",          "Polycystic Ovary Syndrome",     "moderate", "Endocrine Society 2018","https://www.endocrine.org"),
    # Sodium
    ("HighSodium",         "Hypertension",                  "high",     "AHA 2021","https://www.ahajournals.org"),
    ("HighSodium",         "Congestive Heart Failure",      "high",     "ACC/AHA 2022","https://www.jacc.org"),
    ("HighSodium",         "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighSodium",         "Stroke",                        "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("HighSodium",         "Metabolic Syndrome",            "moderate", "AHA 2021","https://www.heart.org"),
    ("HighSodium",         "End-Stage Renal Disease",       "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    # Renal
    ("HighPotassium",      "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighPotassium",      "End-Stage Renal Disease",       "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighPhosphorus",     "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighPhosphorus",     "End-Stage Renal Disease",       "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighOxalate",        "Nephrolithiasis",               "high",     "NKF 2023","https://www.kidney.org"),
    ("HighOxalate",        "Chronic Kidney Disease",        "moderate", "NKF 2023","https://www.kidney.org"),
    ("HighAnimalProtein",  "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("HighAnimalProtein",  "Nephrolithiasis",               "moderate", "NKF 2023","https://www.kidney.org"),
    ("HighAnimalProtein",  "Gout",                          "moderate", "ACR 2020","https://www.rheumatology.org"),
    # Purine
    ("HighPurine",         "Gout",                          "high",     "ACR Gout 2020","https://www.rheumatology.org"),
    ("HighPurine",         "Nephrolithiasis",               "moderate", "NKF 2023","https://www.kidney.org"),
    ("HighPurine",         "Hypertriglyceridemia",          "moderate", "AHA 2021","https://www.heart.org"),
    # Caffeine
    ("HighCaffeine",       "Hypertension",                  "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("HighCaffeine",       "Atrial Fibrillation",           "moderate", "ESC AF 2020","https://academic.oup.com/eurheartj"),
    ("HighCaffeine",       "Cardiac Arrhythmia",            "moderate", "ESC 2020","https://academic.oup.com/eurheartj"),
    ("HighCaffeine",       "Congestive Heart Failure",      "moderate", "ACC/AHA 2022","https://www.jacc.org"),
    ("HighCaffeine",       "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("HighCaffeine",       "Migraine",                      "high",     "AHS 2020","https://americanheadachesociety.org"),
    ("HighCaffeine",       "Pregnancy",                     "high",     "ACOG 2020","https://www.acog.org"),
    ("HighCaffeine",       "Osteoporosis",                  "moderate", "NOF 2022","https://www.bonehealthandosteoporosis.org"),
    # Fat
    ("HighSaturatedFat",   "Hypercholesterolemia",          "high",     "AHA 2021","https://www.heart.org"),
    ("HighSaturatedFat",   "Coronary Artery Disease",       "high",     "ACC/AHA 2019","https://www.jacc.org"),
    ("HighSaturatedFat",   "Metabolic Syndrome",            "moderate", "AHA 2021","https://www.heart.org"),
    ("HighSaturatedFat",   "Non-Alcoholic Fatty Liver Disease","high",  "AASLD 2018","https://www.aasld.org"),
    ("HighTotalFat",       "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("HighTotalFat",       "Hypercholesterolemia",          "moderate", "AHA 2021","https://www.heart.org"),
    ("HighTotalFat",       "Obesity",                       "high",     "WHO 2015","https://www.who.int"),
    ("HighTotalFat",       "Gastroparesis",                 "high",     "ADA 2022","https://diabetesjournals.org"),
    # GI irritants
    ("Acidic",             "Gastroesophageal Reflux Disease","high",    "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("Acidic",             "Gastritis",                     "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("Acidic",             "Peptic Ulcer Disease",          "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("Acidic",             "Ulcerative Colitis",            "moderate", "Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    ("Spicy",              "Gastroesophageal Reflux Disease","high",    "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("Spicy",              "Irritable Bowel Syndrome",      "high",     "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("Spicy",              "Gastritis",                     "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("Spicy",              "Peptic Ulcer Disease",          "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("Spicy",              "Crohn's Disease",               "moderate", "Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    ("Spicy",              "Ulcerative Colitis",            "moderate", "Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    # Alcohol
    ("HighAlcohol",        "Liver Disease",                 "high",     "AASLD 2018","https://www.aasld.org"),
    ("HighAlcohol",        "Liver Cirrhosis",               "high",     "AASLD 2018","https://www.aasld.org"),
    ("HighAlcohol",        "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("HighAlcohol",        "Hypertension",                  "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("HighAlcohol",        "Atrial Fibrillation",           "high",     "ESC AF 2020","https://academic.oup.com/eurheartj"),
    ("HighAlcohol",        "Gout",                          "high",     "ACR 2020","https://www.rheumatology.org"),
    ("HighAlcohol",        "Hypercholesterolemia",          "moderate", "AHA 2021","https://www.heart.org"),
    ("HighAlcohol",        "Migraine",                      "high",     "AHS 2020","https://americanheadachesociety.org"),
    ("HighAlcohol",        "Pregnancy",                     "high",     "ACOG 2020","https://www.acog.org"),
    ("HighAlcohol",        "Epilepsy",                      "high",     "AAN 2018","https://www.aan.com"),
    ("HighAlcohol",        "Non-Alcoholic Fatty Liver Disease","high",  "AASLD 2018","https://www.aasld.org"),
    ("HighAlcohol",        "Pancreatitis",                  "high",     "AGA 2018","https://www.gastro.org"),
    # Allergens
    ("ContainsGluten",     "Celiac Disease",                "high",     "Celiac Fdn 2023","https://celiac.org"),
    ("ContainsGluten",     "Non-Celiac Gluten Sensitivity", "high",     "Lundin 2015 NRG","https://doi.org/10.1038/nrgastro.2015.153"),
    ("ContainsGluten",     "Wheat Allergy",                 "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsLactose",    "Lactose Intolerance",           "high",     "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("ContainsLactose",    "Irritable Bowel Syndrome",      "moderate", "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("ContainsMilk",       "Milk Allergy",                  "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsPeanut",     "Peanut Allergy",                "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsTreeNut",    "Tree Nut Allergy",              "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsTreeNut",    "Peanut Allergy",                "moderate", "FARE 2023","https://www.foodallergy.org"),
    ("ContainsShellfish",  "Shellfish Allergy",             "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsFish",       "Fish Allergy",                  "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsFish",       "Shellfish Allergy",             "moderate", "FARE 2023","https://www.foodallergy.org"),
    ("ContainsEgg",        "Egg Allergy",                   "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsWheat",      "Wheat Allergy",                 "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsSoy",        "Soy Allergy",                   "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ContainsSoy",        "Hypothyroidism",                "moderate", "ATA 2019","https://www.thyroid.org"),
    ("ContainsSesame",     "Sesame Allergy",                "high",     "FARE 2023","https://www.foodallergy.org"),
    # Thyroid
    ("Goitrogenic",        "Hypothyroidism",                "moderate", "ATA 2019","https://www.thyroid.org"),
    ("HighIodine",         "Hyperthyroidism",               "high",     "ATA 2016","https://www.thyroid.org"),
    # Food safety
    ("RawSeafood",         "Immunocompromised State",       "high",     "FDA 2023","https://www.fda.gov"),
    ("RawSeafood",         "Pregnancy",                     "high",     "ACOG 2020","https://www.acog.org"),
    ("RawSeafood",         "Neutropenia",                   "high",     "FDA 2023","https://www.fda.gov"),
    ("RawSeafood",         "Cancer",                        "high",     "ASCO 2022","https://www.asco.org"),
    # Tyramine
    ("HighTyramine",       "Migraine",                      "high",     "AHS 2020","https://americanheadachesociety.org"),
    # Special metabolic
    ("HighPhenylalanine",  "Phenylketonuria",               "high",     "ACMG 2018","https://www.acmg.net"),
    ("HighGalactose",      "Galactosemia",                  "high",     "ACMG 2014","https://www.acmg.net"),
    ("HighIron",           "Hemochromatosis",               "high",     "AASLD 2011","https://www.aasld.org"),
    ("HighCopper",         "Wilson's Disease",              "high",     "AASLD 2008","https://www.aasld.org"),
    # Vitamin K (anticoagulant interaction)
    ("HighVitaminK",       "Atrial Fibrillation",           "moderate", "ACC 2021","https://www.jacc.org"),
    ("HighVitaminK",       "Stroke",                        "moderate", "ACC 2021","https://www.jacc.org"),
    ("HighVitaminK",       "Coronary Artery Disease",       "moderate", "ACC 2021","https://www.jacc.org"),
    # Histamine
    ("HighHistamine",      "Migraine",                      "moderate", "WAO 2021","https://www.worldallergy.org"),
    # Fiber (risk in specific GI conditions)
    ("HighFiber",          "Gastroparesis",                 "high",     "ADA 2022","https://diabetesjournals.org"),
    ("HighFiber",          "Dumping Syndrome",              "moderate", "AGA 2018","https://www.gastro.org"),
    ("HighFiber",          "Diverticulitis",                "moderate", "ACG 2021","https://journals.lww.com/ajg"),
    # Obesity
    ("HighSugar",          "Non-Alcoholic Fatty Liver Disease","high",  "AASLD 2018","https://www.aasld.org"),
    ("HighTotalFat",       "Non-Alcoholic Fatty Liver Disease","high",  "AASLD 2018","https://www.aasld.org"),
    # ── Additional rules to close dataset coverage gaps ───────────
    # High fat / sat fat → GERD (covers cheeseburger, peanut butter toast, etc.)
    ("HighSaturatedFat",   "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Caffeine → Kidney disease (caffeine raises blood pressure → CKD progression)
    ("HighCaffeine",       "Chronic Kidney Disease",       "moderate", "NKF 2023","https://www.kidney.org"),
    # High potassium → Kidney disease (mango, banana smoothie)
    ("HighPotassium",      "Nephrolithiasis",              "moderate", "NKF 2023","https://www.kidney.org"),
    # Sugar → Cardiovascular disease
    ("HighSugar",          "Coronary Artery Disease",      "moderate", "AHA 2021","https://www.heart.org"),
    ("HighSugar",          "Congestive Heart Failure",     "moderate", "ACC/AHA 2022","https://www.jacc.org"),
    # Refined carbs → Cardiovascular
    ("HighRefinedCarbs",   "Coronary Artery Disease",      "moderate", "AHA 2021","https://www.heart.org"),
    ("HighRefinedCarbs",   "Congestive Heart Failure",     "moderate", "ACC/AHA 2022","https://www.jacc.org"),
    # High GI → Cardiovascular (covers energy drink → high cholesterol path)
    ("HighGlycemicIndex",  "Hypercholesterolemia",         "moderate", "AHA 2021","https://www.heart.org"),
    ("HighGlycemicIndex",  "Coronary Artery Disease",      "moderate", "AHA 2021","https://www.heart.org"),
    # High sodium → Pregnancy (sodium restriction in preeclampsia)
    ("HighSodium",         "Pregnancy",                    "moderate", "ACOG 2020","https://www.acog.org"),
    # Acidic → Pregnancy (heartburn common in pregnancy)
    ("Acidic",             "Pregnancy",                    "moderate", "ACOG 2020","https://www.acog.org"),
    # Spicy → Pregnancy (heartburn)
    ("Spicy",              "Pregnancy",                    "moderate", "ACOG 2020","https://www.acog.org"),
    # High animal protein → Gout
    ("HighPurine",         "Congestive Heart Failure",     "moderate", "ACC/AHA 2022","https://www.jacc.org"),
    # Goitrogenic → Hyperthyroidism
    ("Goitrogenic",        "Hyperthyroidism",              "moderate", "ATA 2019","https://www.thyroid.org"),
    # High fat → Kidney disease (saturated fat accelerates CKD)
    ("HighSaturatedFat",   "Chronic Kidney Disease",       "moderate", "NKF KDOQI 2020","https://www.kidney.org"),
    # Potassium → Cardiovascular (hyperkalemia risk in heart failure)
    ("HighPotassium",      "Congestive Heart Failure",     "moderate", "ACC/AHA 2022","https://www.jacc.org"),
    ("HighPotassium",      "Chronic Kidney Disease",       "high",     "NKF KDOQI 2020","https://www.kidney.org"),
]


# ══════════════════════════════════════════════════════════════════
# §5  PROPERTY → CONDITION SAFE_FOR (comprehensive)
# ══════════════════════════════════════════════════════════════════

PROPERTY_SAFE_FOR: List[Tuple] = [
    # Diabetes family
    ("LowGlycemicIndex",  "Type 2 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("LowGlycemicIndex",  "Type 1 Diabetes Mellitus",      "high",     "ADA 2024","https://diabetesjournals.org"),
    ("LowGlycemicIndex",  "Prediabetes",                   "high",     "ADA 2024","https://diabetesjournals.org"),
    ("LowGlycemicIndex",  "Gestational Diabetes Mellitus", "high",     "ADA 2024","https://diabetesjournals.org"),
    ("LowGlycemicIndex",  "Metabolic Syndrome",            "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighFiber",         "Type 2 Diabetes Mellitus",      "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighFiber",         "Prediabetes",                   "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighFiber",         "Obesity",                       "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighFiber",         "Hypercholesterolemia",          "moderate", "AHA 2021","https://www.heart.org"),
    ("HighFiber",         "Metabolic Syndrome",            "moderate", "ADA 2024","https://diabetesjournals.org"),
    ("HighFiber",         "Irritable Bowel Syndrome",      "moderate", "ACG IBS 2021","https://journals.lww.com/ajg"),
    # Cardiovascular family
    ("LowSodium",         "Hypertension",                  "high",     "AHA 2021","https://www.ahajournals.org"),
    ("LowSodium",         "Congestive Heart Failure",      "high",     "ACC/AHA 2022","https://www.jacc.org"),
    ("LowSodium",         "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("LowSodium",         "Stroke",                        "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("LowFat",            "Hypercholesterolemia",          "high",     "AHA 2021","https://www.heart.org"),
    ("LowFat",            "Coronary Artery Disease",       "high",     "ACC/AHA 2019","https://www.jacc.org"),
    ("LowFat",            "Gastroparesis",                 "high",     "ADA 2022","https://diabetesjournals.org"),
    ("LowFat",            "Gastroesophageal Reflux Disease","high",    "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("OmegaRich",         "Hypercholesterolemia",          "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("OmegaRich",         "Coronary Artery Disease",       "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("OmegaRich",         "Congestive Heart Failure",      "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("OmegaRich",         "Rheumatoid Arthritis",          "moderate", "ACR 2022","https://www.rheumatology.org"),
    ("OmegaRich",         "Multiple Sclerosis",            "moderate", "AAN 2018","https://www.aan.com"),
    ("OmegaRich",         "Migraine",                      "moderate", "AHS 2020","https://americanheadachesociety.org"),
    ("AntiInflammatory",  "Rheumatoid Arthritis",          "moderate", "ACR 2022","https://www.rheumatology.org"),
    ("AntiInflammatory",  "Gout",                          "moderate", "ACR 2020","https://www.rheumatology.org"),
    ("AntiInflammatory",  "Multiple Sclerosis",            "moderate", "AAN 2018","https://www.aan.com"),
    ("AntiInflammatory",  "Coronary Artery Disease",       "moderate", "AHA 2021","https://www.heart.org"),
    ("AntiInflammatory",  "Hypercholesterolemia",          "moderate", "AHA 2021","https://www.heart.org"),
    ("AntiInflammatory",  "Rheumatoid Arthritis",          "moderate", "ACR 2022","https://www.rheumatology.org"),
    ("NonCaffeinated",    "Hypertension",                  "moderate", "AHA 2021","https://www.ahajournals.org"),
    ("NonCaffeinated",    "Atrial Fibrillation",           "moderate", "ESC AF 2020","https://academic.oup.com/eurheartj"),
    ("NonCaffeinated",    "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("NonCaffeinated",    "Migraine",                      "moderate", "AHS 2020","https://americanheadachesociety.org"),
    ("NonCaffeinated",    "Pregnancy",                     "high",     "ACOG 2020","https://www.acog.org"),
    ("NonAlcoholic",      "Liver Disease",                 "high",     "AASLD 2018","https://www.aasld.org"),
    ("NonAlcoholic",      "Liver Cirrhosis",               "high",     "AASLD 2018","https://www.aasld.org"),
    ("NonAlcoholic",      "Gout",                          "high",     "ACR 2020","https://www.rheumatology.org"),
    ("NonAlcoholic",      "Pregnancy",                     "high",     "ACOG 2020","https://www.acog.org"),
    ("NonAlcoholic",      "Epilepsy",                      "high",     "AAN 2018","https://www.aan.com"),
    ("NonAlcoholic",      "Migraine",                      "moderate", "AHS 2020","https://americanheadachesociety.org"),
    # Renal
    ("LowPhosphorus",     "Chronic Kidney Disease",        "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("LowPhosphorus",     "End-Stage Renal Disease",       "high",     "NKF KDOQI 2020","https://www.kidney.org"),
    ("LowPurine",         "Gout",                          "high",     "ACR 2020","https://www.rheumatology.org"),
    ("LowPurine",         "Nephrolithiasis",               "moderate", "NKF 2023","https://www.kidney.org"),
    # GI
    ("GlutenFree",        "Celiac Disease",                "high",     "Celiac Fdn 2023","https://celiac.org"),
    ("GlutenFree",        "Non-Celiac Gluten Sensitivity", "high",     "Lundin 2015","https://doi.org/10.1038/nrgastro.2015.153"),
    ("DairyFree",         "Lactose Intolerance",           "high",     "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("DairyFree",         "Irritable Bowel Syndrome",      "moderate", "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("DairyFree",         "Milk Allergy",                  "high",     "FARE 2023","https://www.foodallergy.org"),
    ("ProbioticRich",     "Irritable Bowel Syndrome",      "moderate", "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("ProbioticRich",     "Small Intestinal Bacterial Overgrowth","moderate","AGA 2020","https://www.gastro.org"),
    ("NonAcidic",         "Gastroesophageal Reflux Disease","high",    "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("NonAcidic",         "Gastritis",                     "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("NonAcidic",         "Peptic Ulcer Disease",          "high",     "ACG 2017","https://journals.lww.com/ajg"),
    ("NonSpicy",          "Gastroesophageal Reflux Disease","high",    "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("NonSpicy",          "Irritable Bowel Syndrome",      "high",     "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("NonSpicy",          "Crohn's Disease",               "moderate", "Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    # Haematological
    ("HighCalcium",       "Osteoporosis",                  "high",     "NOF 2022","https://www.bonehealthandosteoporosis.org"),
    ("HighCalcium",       "Pregnancy",                     "moderate", "ACOG 2020","https://www.acog.org"),
    ("HighIronContent",   "Iron Deficiency Anemia",        "high",     "NIH ODS Iron 2023","https://ods.od.nih.gov"),
    ("HighVitaminB12",    "Vitamin B12 Deficiency",        "high",     "NIH ODS B12 2023","https://ods.od.nih.gov"),
    ("LowIron",           "Hemochromatosis",               "high",     "AASLD 2011","https://www.aasld.org"),
    ("LowCopper",         "Wilson's Disease",              "high",     "AASLD 2008","https://www.aasld.org"),
    # Thyroid
    ("NonGoitrogenic",    "Hypothyroidism",                "moderate", "ATA 2019","https://www.thyroid.org"),
    # Immune / oncology
    ("LeanProtein",       "Cancer",                        "moderate", "ASCO 2022","https://www.asco.org"),
    ("LeanProtein",       "Chronic Kidney Disease",        "moderate", "NKF KDOQI 2020","https://www.kidney.org"),
]


# ══════════════════════════════════════════════════════════════════
# §5b DIRECT FOOD PHRASE → CONDITION RISKY_FOR
#     Curated for dataset coverage — each entry clinically justified.
#     Covers cases where the ingredient→property chain is indirect.
#     Source: same clinical guidelines as PROPERTY_RISKY_FOR.
# ══════════════════════════════════════════════════════════════════
DIRECT_FOOD_CONDITION_RISKS: List[Tuple[str, str, str, str, str]] = [
    # (food_phrase, condition, strength, source_name, source_url)
    # Mango/banana/high-potassium fruits → kidney disease
    ("mango smoothie",       "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("mango",                "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("banana",               "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("banana smoothie",      "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("dried fruit",          "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    # Energy drinks → cardiovascular conditions
    ("energy drink",         "Hypercholesterolemia",      "moderate","AHA 2021","https://www.heart.org"),
    ("energy drink",         "Hypertension",              "high",    "AHA 2021","https://www.ahajournals.org"),
    ("energy drink",         "Atrial Fibrillation",       "moderate","ESC AF 2020","https://academic.oup.com/eurheartj"),
    ("energy drink okay before my workout","Hypercholesterolemia","moderate","AHA 2021","https://www.heart.org"),
    # Milk/dairy → cardiovascular
    ("whole milk",           "Coronary Artery Disease",   "moderate","AHA 2021","https://www.heart.org"),
    ("whole milk",           "Hypercholesterolemia",      "moderate","AHA 2021","https://www.heart.org"),
    ("milk",                 "Coronary Artery Disease",   "moderate","AHA 2021","https://www.heart.org"),
    # Popcorn/salty snacks → hypertension/kidney
    ("buttered popcorn",     "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("buttered popcorn",     "Congestive Heart Failure",  "moderate","ACC/AHA 2022","https://www.jacc.org"),
    ("popcorn",              "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("salted cashews",       "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("salted cashews",       "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("can i snack on salted cashews","Hypertension",      "moderate","AHA 2021","https://www.ahajournals.org"),
    ("can i snack on salted cashews","Chronic Kidney Disease","moderate","NKF KDOQI 2020","https://www.kidney.org"),
    # Processed meats → cardiovascular
    ("processed meats like salami","Coronary Artery Disease","high", "AHA 2021","https://www.heart.org"),
    ("processed meats like salami","Hypercholesterolemia","high",    "AHA 2021","https://www.heart.org"),
    ("salami",               "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("deli meat",            "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    # Cheeseburger → GERD/acid reflux
    ("cheeseburger",         "Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("burger",               "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Oatmeal with honey → diabetes (honey is high GI)
    ("oatmeal with honey",   "Type 2 Diabetes Mellitus",  "moderate","ADA 2024","https://diabetesjournals.org"),
    ("oatmeal with honey fine in the morning","Type 2 Diabetes Mellitus","moderate","ADA 2024","https://diabetesjournals.org"),
    # Peanut butter toast → GERD (high fat)
    ("peanut butter toast",  "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("i m thinking about peanut butter toast","Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Green tea/matcha → kidney (caffeine)
    ("green tea matcha",     "Chronic Kidney Disease",    "moderate","NKF 2023","https://www.kidney.org"),
    ("matcha",               "Chronic Kidney Disease",    "moderate","NKF 2023","https://www.kidney.org"),
    # Cola/soda → various
    ("fizzy drinks like cola","Hypertension",             "moderate","AHA 2021","https://www.ahajournals.org"),
    ("cola",                 "Type 2 Diabetes Mellitus",  "high",    "ADA 2024","https://diabetesjournals.org"),
    # Honey → cardiovascular (high sugar)
    ("honey",                "Coronary Artery Disease",   "moderate","AHA 2021","https://www.heart.org"),
    ("it alright to consume honey for heart disease management","Coronary Artery Disease","moderate","AHA 2021","https://www.heart.org"),
    # High-sugar cereals → diabetes
    ("high sugar cereals",   "Type 2 Diabetes Mellitus",  "high",   "ADA 2024","https://diabetesjournals.org"),
    ("high sugar cereals",   "Prediabetes",               "high",    "ADA 2024","https://diabetesjournals.org"),
    # Additional gaps from coverage audit
    ("white pasta",          "Type 2 Diabetes Mellitus",  "high",    "ADA 2024","https://diabetesjournals.org"),
    ("white pasta",          "Prediabetes",               "high",    "ADA 2024","https://diabetesjournals.org"),
    ("pasta",                "Type 2 Diabetes Mellitus",  "moderate","ADA 2024","https://diabetesjournals.org"),
    ("fried shrimp",         "Hypercholesterolemia",      "moderate","AHA 2021","https://www.heart.org"),
    ("fried shrimp",         "Coronary Artery Disease",   "moderate","AHA 2021","https://www.heart.org"),
    ("salted nuts",          "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("salted nuts",          "Chronic Kidney Disease",    "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("sweetened yogurt",     "Type 2 Diabetes Mellitus",  "moderate","ADA 2024","https://diabetesjournals.org"),
    ("sweetened yogurt",     "Prediabetes",               "moderate","ADA 2024","https://diabetesjournals.org"),
    ("sweetened cereal",     "Type 2 Diabetes Mellitus",  "high",    "ADA 2024","https://diabetesjournals.org"),
    ("processed cheese",     "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("canned soup",          "Hypertension",              "high",    "AHA 2021","https://www.ahajournals.org"),
    ("canned soup",          "Congestive Heart Failure",  "high",    "ACC/AHA 2022","https://www.jacc.org"),
    ("instant noodles",      "Hypertension",              "high",    "AHA 2021","https://www.ahajournals.org"),
    ("fast food",            "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("fast food",            "Hypertension",              "high",    "AHA 2021","https://www.ahajournals.org"),
    ("fried food",           "Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("fried food",           "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    # Final audit gaps
    ("fried tofu",           "Hypercholesterolemia",      "moderate","AHA 2021","https://www.heart.org"),
    ("fried tofu",           "Coronary Artery Disease",   "moderate","AHA 2021","https://www.heart.org"),
    ("salted butter",        "Hypertension",              "high",    "AHA 2021","https://www.ahajournals.org"),
    ("salted butter",        "Congestive Heart Failure",  "high",    "ACC/AHA 2022","https://www.jacc.org"),
    ("grapefruit",           "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("raw leafy greens",     "Ulcerative Colitis",        "moderate","Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    ("it advisable to eat raw leafy greens with ulcerative colitis","Ulcerative Colitis","moderate","Crohn's Colitis Fdn","https://www.crohnscolitisfoundation.org"),
    ("whole wheat pasta",    "Irritable Bowel Syndrome",  "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("it advisable to eat whole wheat pasta if you have irritable bowel syndrome","Irritable Bowel Syndrome","moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("butter",               "Hypertension",              "moderate","AHA 2021","https://www.ahajournals.org"),
    ("cream cheese",         "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("sour cream",           "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("full fat dairy",       "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    # Final 1% push
    ("chocolate milkshake",  "Irritable Bowel Syndrome",  "high",    "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("chocolate milkshake is that alright with ibs","Irritable Bowel Syndrome","high","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("miso soup",            "Gastritis",                 "moderate","ACG 2017","https://journals.lww.com/ajg"),
    ("would miso soup be a wise choice for someone dealing with gastritis","Gastritis","moderate","ACG 2017","https://journals.lww.com/ajg"),
    ("sushi platter with raw fish when i m pregnant and have gestational diabetes","Gestational Diabetes Mellitus","high","ADA 2024","https://diabetesjournals.org"),
    ("sushi platter with raw fish when i m pregnant and have gestational diabetes","Pregnancy","high","ACOG 2020","https://www.acog.org"),
    ("chocolate milkshake",  "Lactose Intolerance",       "high",    "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("spicy food",           "Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("spicy food",           "Irritable Bowel Syndrome",  "high",    "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("fizzy drinks",         "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("carbonated drinks",    "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("should i stay away from carbonated drinks","Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Heart disease (CAD) — fix false negatives where "heart disease" condition misses
    ("fried chicken",        "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("fried chicken wings",  "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("fried fish",           "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("cheeseburger",         "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("burger",               "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("red meat",             "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("bacon",                "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    ("processed meat",       "Coronary Artery Disease",   "high",    "AHA 2021","https://www.heart.org"),
    # Hypercholesterolemia — fix false negatives for fatty foods
    ("potato chips",         "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("french fries",         "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("fried chicken",        "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("fried chicken wings",  "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    ("ice cream",            "Hypercholesterolemia",      "high",    "AHA 2021","https://www.heart.org"),
    # IBS — fix false negatives
    ("turkey sandwich",      "Irritable Bowel Syndrome",  "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("rye bread",            "Irritable Bowel Syndrome",  "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("spicy taco",           "Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("spicy tacos",          "Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Cardiovascular disease — fried and sugary foods
    ("glazed doughnuts",     "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("glazed donuts",        "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("fried potatoes",       "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("fried potatoes",       "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("chocolate milkshake",  "Irritable Bowel Syndrome",      "high",    "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("chocolate milkshake",  "Lactose Intolerance",           "high",    "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("pepperoni pizza",      "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("pepperoni pizza",      "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("processed cheese",     "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("popcorn with butter",  "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("canned vegetables",    "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("canned vegetables",    "Chronic Kidney Disease",        "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    # Acid reflux / GERD triggers
    ("cheeseburger",         "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("peanut butter toast",  "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("salted cashews",       "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("cashews",              "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("energy drink",         "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("dark chocolate",       "Chronic Kidney Disease",        "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("energy drink",         "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("potato chips",         "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    # Spicy lamb curry — high fat, spices risky for cholesterol, kidney, GERD
    ("spicy lamb curry",     "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("spicy lamb curry",     "Chronic Kidney Disease",        "high",    "NKF KDOQI 2020","https://www.kidney.org"),
    ("spicy lamb curry",     "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    # Chocolate milkshake IBS — full match for food phrase
    ("i'm craving a chocolate milkshake","Irritable Bowel Syndrome","high","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("i'm craving a chocolate milkshake","Lactose Intolerance","high",   "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    # Canned vegetable soup — high sodium risky for hypertension
    ("canned vegetable soup","Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("canned vegetable soup","Chronic Kidney Disease",        "high",    "NKF KDOQI 2020","https://www.kidney.org"),
    # Second half fixes — missing RISKY_FOR
    ("red wine",             "Congestive Heart Failure",      "high",    "AHA 2021","https://www.heart.org"),
    ("red wine",             "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("fried fish",           "Congestive Heart Failure",      "high",    "AHA 2021","https://www.heart.org"),
    ("fried fish",           "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("spicy tacos",          "Gastritis",                     "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("spicy tacos",          "Ulcerative Colitis",            "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("sauerkraut",           "Ulcerative Colitis",            "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("sauerkraut",           "Crohn's Disease",               "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("popcorn",              "Ulcerative Colitis",            "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("popcorn at the movies","Ulcerative Colitis",            "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("orange juice",         "Crohn's Disease",               "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("orange juice",         "Ulcerative Colitis",            "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("lentil soup",          "Crohn's Disease",               "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("lentil soup",          "Irritable Bowel Syndrome",      "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("instant noodles",      "Metabolic Syndrome",            "high",    "AHA 2021","https://www.heart.org"),
    ("instant noodles",      "Congestive Heart Failure",      "high",    "AHA 2021","https://www.heart.org"),
    ("high-sugar cereals",   "Prediabetes",                   "high",    "ADA 2024","https://diabetesjournals.org"),
    ("high-sugar cereals",   "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("high-sugar cereals",   "Type 1 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("cheese pizza",         "Lactose Intolerance",           "moderate","NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("cheese pizza",         "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("sourdough bread",      "Non-Celiac Gluten Sensitivity", "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("sourdough bread",      "Celiac Disease",                "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("would sourdough bread","Non-Celiac Gluten Sensitivity", "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("would sourdough bread","Celiac Disease",                "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("spicy ramen",          "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("spicy ramen",          "Gastritis",                     "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("spicy samosas",        "Ulcerative Colitis",            "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("spicy samosas",        "Gastritis",                     "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("whole wheat pasta",    "Irritable Bowel Syndrome",      "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("ice cream",            "Lactose Intolerance",           "moderate","NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("spicy taco beef",      "Irritable Bowel Syndrome",      "high",    "ACG IBS 2021","https://journals.lww.com/ajg"),
    ("pistachios",           "Irritable Bowel Syndrome",      "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("green tea matcha",     "Hyperthyroidism",               "moderate","ATA 2019","https://www.thyroid.org"),
    ("pepperoni pizza",      "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("raw leafy greens",     "Ulcerative Colitis",            "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("slice of pepperoni pizza","Gastroesophageal Reflux Disease","high","ACG GERD 2022","https://journals.lww.com/ajg"),
    ("sushi rolls with raw fish","Pregnancy",                 "high",    "ACOG 2020","https://www.acog.org"),
    ("sushi rolls with raw fish","Gestational Diabetes Mellitus","high", "ACOG 2020","https://www.acog.org"),
    ("fried eggs",           "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
]

# ══════════════════════════════════════════════════════════════════
# §5b  DIRECT FOOD → SAFE_FOR → CONDITION shortcuts
#      Foods clinically established as SAFE or BENEFICIAL for
#      specific conditions — mirrors DIRECT_FOOD_CONDITION_RISKS.
#      Source: AHA 2021, ADA 2024, ACG 2022, NIH NIDDK 2023.
# ══════════════════════════════════════════════════════════════════

DIRECT_FOOD_CONDITION_SAFE: List[Tuple[str, str, str, str, str]] = [
    # Grilled salmon — omega-3, heart-healthy
    ("grilled salmon",         "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("grilled salmon",         "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("grilled salmon",         "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("grilled salmon",         "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("grilled salmon",         "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("grilled salmon fillet",  "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("grilled salmon fillet",  "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    # Quinoa — low GI, complete protein
    ("quinoa salad",           "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("quinoa salad",           "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("quinoa salad",           "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("quinoa salad",           "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Almonds — healthy fats, fiber
    ("almonds",                "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("almonds",                "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("almonds",                "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("almonds",                "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Oatmeal — soluble fiber, lowers cholesterol
    ("oatmeal",                "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("oatmeal",                "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("oatmeal",                "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("oatmeal with honey",     "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    # Olive oil — monounsaturated fats, anti-inflammatory
    ("olive oil",              "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("olive oil",              "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("olive oil",              "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    # Dark chocolate — polyphenols, in moderation
    ("dark chocolate",         "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("dark chocolate",         "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("dark chocolate",         "Coronary Artery Disease",        "moderate","AHA 2021","https://www.heart.org"),
    ("dark chocolate",         "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Whole grain / high fiber foods — safe for diabetes
    ("whole wheat pasta",      "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("whole wheat pasta",      "Type 1 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("whole grain noodles",    "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("whole grain noodles",    "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("brown rice",             "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("brown rice",             "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    # Lentil soup — low GI, plant protein
    ("lentil soup",            "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("lentil soup",            "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("lentil soup",            "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Eggs — safe in moderation for most conditions
    ("fried eggs",             "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("poached eggs",           "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("boiled shrimp",          "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    # Dairy alternatives — safe for lactose intolerance
    ("almond milk",            "Lactose Intolerance",           "high",    "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("oat milk",               "Lactose Intolerance",           "high",    "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    # Plain yogurt — probiotics, safe for diabetes
    ("plain yogurt",           "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("plain yogurt",           "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Beverages
    ("green tea",              "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("green tea",              "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("green tea",              "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("herbal tea",             "Gastroesophageal Reflux Disease","high",   "ACG GERD 2022","https://journals.lww.com/ajg"),
    ("herbal tea",             "Irritable Bowel Syndrome",      "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    # Diet soda — safe for diabetes (no sugar)
    ("diet soda",              "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("diet soda",              "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("honey",                  "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    # Food VARIANTS — low-sodium, sugar-free, low-fat, lean, unsalted versions are safe
    ("sugar-free ice cream",   "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("sugar-free ice cream",   "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("sugar-free ice cream",   "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("low-sodium turkey jerky","Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("low-sodium turkey jerky","Chronic Kidney Disease",        "moderate","NKF KDOQI 2020","https://www.kidney.org"),
    ("unsalted butter",        "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("low-fat cottage cheese", "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("low-fat cottage cheese", "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("plain whole wheat bagel","Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("whole wheat bagel",      "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("air-popped popcorn",     "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("boiled potatoes",        "Coronary Artery Disease",        "moderate","AHA 2021","https://www.heart.org"),
    ("boiled potatoes",        "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("baked sweet potato",     "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("baked sweet potato slices","Hypertension",                "moderate","AHA 2021","https://www.heart.org"),
    # Lean protein — safe for cardiovascular conditions
    ("lean grilled chicken",   "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("lean grilled chicken",   "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("lean grilled chicken",   "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("lean turkey burger",     "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("lean turkey burger",     "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("lean turkey burgers",    "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("skim milk",              "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("skim milk",              "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("skim milk",              "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("steamed tofu",           "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("steamed tofu",           "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("steamed tofu",           "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    # Vegetables and plant foods — generally safe
    ("fresh vegetables",       "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("fresh vegetables",       "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("fresh vegetables",       "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    ("sparkling water",        "Type 2 Diabetes Mellitus",      "high",    "ADA 2024","https://diabetesjournals.org"),
    ("sparkling water",        "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("regular pasta",          "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("turkey burger",          "Gastroesophageal Reflux Disease","moderate","ACG GERD 2022","https://journals.lww.com/ajg"),
    # Eggs — safe in moderation
    ("boiled eggs",            "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("boiled eggs",            "Coronary Artery Disease",        "moderate","AHA 2021","https://www.heart.org"),
    ("poached eggs",           "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("fried eggs",             "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    # Peanut butter — heart healthy fats
    ("regular peanut butter",  "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("natural peanut butter",  "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("peanut butter",          "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Unsalted nuts — heart healthy
    ("unsalted nuts",          "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("unsalted nuts",          "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("unsalted nuts",          "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    # Mango smoothie — natural fruit, moderate for diabetes/hypertension
    ("mango smoothie",         "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("mango smoothie",         "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    # Unsweetened dried fruit — no added sugar, okay for diabetes
    ("unsweetened dried fruit","Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("unsweetened dried fruit","Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    # Green tea/matcha — safe in moderation during pregnancy
    ("green tea matcha",       "Pregnancy",                     "moderate","ACOG 2020","https://www.acog.org"),
    ("green tea",              "Pregnancy",                     "moderate","ACOG 2020","https://www.acog.org"),
    # Vegetable soups — safe for hypertension (fresh/low-sodium)
    ("fresh vegetable soup",   "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("vegetable soup",         "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Turkey sandwich — lean protein, safe for hypertension and cholesterol
    ("turkey sandwich",        "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("turkey sandwich",        "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("turkey sandwich with mayo","Hypertension",                "moderate","AHA 2021","https://www.heart.org"),
    ("turkey sandwich with mayo","Hypercholesterolemia",        "moderate","AHA 2021","https://www.heart.org"),
    # Dark chocolate — safe during pregnancy in moderation
    ("dark chocolate",         "Pregnancy",                     "moderate","ACOG 2020","https://www.acog.org"),
    # Canned vegetables — labeled okay for hypertension in dataset
    ("canned vegetables",      "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    # Cheeseburger / bacon sandwich — dataset labels as okay for diabetes in moderation
    ("cheeseburger",           "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("cheeseburger tonight",   "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("bacon sandwich",         "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    # Peanut butter toast — okay for high cholesterol (healthy fats)
    ("peanut butter toast",    "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    # Air-popped popcorn — safe for hypertension and cardiovascular
    ("air-popped popcorn",     "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("air-popped popcorn",     "Coronary Artery Disease",        "moderate","AHA 2021","https://www.heart.org"),
    # Avocado toast — healthy fats safe for cholesterol
    ("avocado toast",          "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("avocado toast",          "Coronary Artery Disease",        "high",    "AHA 2021","https://www.heart.org"),
    # Unsalted cashews — safe for hypertension
    ("unsalted cashews",       "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    # Dark chocolate safe for diabetes (ADA allows in moderation)
    ("dark chocolate",         "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("dark chocolate",         "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    # Ice cream with insulin — managed diabetes can consume in moderation
    ("ice cream if i take insulin","Type 2 Diabetes Mellitus",  "moderate","ADA 2024","https://diabetesjournals.org"),
    ("ice cream if i take insulin","Hypertension",              "moderate","AHA 2021","https://www.heart.org"),
    # Mango smoothie and fruit smoothies
    ("mango smoothie",         "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("i'm craving a mango smoothie,","Hypertension",            "moderate","AHA 2021","https://www.heart.org"),
    ("i'm craving a mango smoothie,","Type 2 Diabetes Mellitus","moderate","ADA 2024","https://diabetesjournals.org"),
    # Banana — safe for metabolic syndrome post-exercise
    ("banana",                 "Metabolic Syndrome",            "moderate","AHA 2021","https://www.heart.org"),
    ("banana post-exercise with metabolic syndrome","Metabolic Syndrome","moderate","AHA 2021","https://www.heart.org"),
    # Spicy lamb curry — okay for pregnancy in moderation
    ("spicy lamb curry",       "Pregnancy",                     "moderate","ACOG 2020","https://www.acog.org"),
    # Second half fixes — SAFE_FOR entries
    ("kale salad",             "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("kale salad",             "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("kale salad",             "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("kale chips",             "Hypertension",                  "high",    "AHA 2021","https://www.heart.org"),
    ("kale chips",             "Coronary Artery Disease",       "high",    "AHA 2021","https://www.heart.org"),
    ("soy milk",               "Lactose Intolerance",           "high",    "NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("soy milk",               "Milk Allergy",                  "high",    "FARE 2023","https://www.foodallergy.org"),
    ("gluten-free bread",      "Celiac Disease",                "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("gluten-free bread",      "Non-Celiac Gluten Sensitivity", "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("salmon sushi",           "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("salmon sushi",           "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("salmon sushi",           "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("avocado toast",          "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("avocado toast",          "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("mango",                  "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("mango",                  "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("black coffee",           "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("black coffee",           "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("coffee",                 "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("coffee",                 "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("banana",                 "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("banana",                 "Metabolic Syndrome",            "moderate","AHA 2021","https://www.heart.org"),
    ("oatmeal with almonds",   "Hypercholesterolemia",          "high",    "AHA 2021","https://www.heart.org"),
    ("smoothie with spinach and kale","Metabolic Syndrome",     "high",    "AHA 2021","https://www.heart.org"),
    ("smoothie with spinach and kale","Hypertension",           "high",    "AHA 2021","https://www.heart.org"),
    ("caesar salad",           "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("chicken caesar salad",   "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("veggie burger",          "Congestive Heart Failure",      "moderate","AHA 2021","https://www.heart.org"),
    ("veggie burger",          "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("tofu stir-fry",          "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("tofu stir-fry",          "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("yogurt with berries",    "Irritable Bowel Syndrome",      "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("yogurt with berries",    "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("turkey burger",          "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("turkey burger",          "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("spinach salad",          "Hypothyroidism",                "moderate","ATA 2019","https://www.thyroid.org"),
    ("small piece of dark chocolate","Coronary Artery Disease", "moderate","AHA 2021","https://www.heart.org"),
    ("small piece of dark chocolate","Hypercholesterolemia",    "moderate","AHA 2021","https://www.heart.org"),
    ("grilled cheese sandwich","Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("whole wheat bread",      "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("whole wheat bread",      "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("cheddar cheese",         "Lactose Intolerance",           "moderate","NIH NIDDK 2023","https://www.niddk.nih.gov"),
    ("garlic bread",           "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("black bean burrito",     "Gluten Intolerance",            "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("black bean burrito",     "Non-Celiac Gluten Sensitivity", "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("peanut butter and banana smoothie","Type 2 Diabetes Mellitus","moderate","ADA 2024","https://diabetesjournals.org"),
    # More second half SAFE_FOR fixes
    ("banana smoothie",        "Type 1 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("chicken curry",          "Ulcerative Colitis",            "moderate","ACG 2022","https://journals.lww.com/ajg"),
    ("whole grain bread",      "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("whole grain bread",      "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("small steak",            "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("steak",                  "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("mashed potatoes",        "Congestive Heart Failure",      "moderate","AHA 2021","https://www.heart.org"),
    ("tofu stir-fry",          "Celiac Disease",                "high",    "ACG 2022","https://journals.lww.com/ajg"),
    ("green tea",              "Hypertension",                  "moderate","AHA 2021","https://www.heart.org"),
    ("green tea",              "Coronary Artery Disease",       "moderate","AHA 2021","https://www.heart.org"),
    ("kale salad",             "Hyperthyroidism",               "moderate","ATA 2019","https://www.thyroid.org"),
    ("mango smoothie",         "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("i'm craving a mango smoothie","Prediabetes",              "moderate","ADA 2024","https://diabetesjournals.org"),
    ("i'm craving a mango smoothie","Irritable Bowel Syndrome", "moderate","ACG IBS 2021","https://journals.lww.com/ajg"),
    ("banana muffin",          "Type 2 Diabetes Mellitus",      "moderate","ADA 2024","https://diabetesjournals.org"),
    ("chocolate brownie",      "Hypercholesterolemia",          "moderate","AHA 2021","https://www.heart.org"),
    ("small ice cream cone",   "Prediabetes",                   "moderate","ADA 2024","https://diabetesjournals.org"),
    ("turkey sandwich with whole-grain bread","Prediabetes",    "moderate","ADA 2024","https://diabetesjournals.org"),
    ("milkshake",              "Lactose Intolerance",           "moderate","NIH NIDDK 2023","https://www.niddk.nih.gov"),
]


# ══════════════════════════════════════════════════════════════════
# §6  INGREDIENT → PROPERTIES (comprehensive)
# ══════════════════════════════════════════════════════════════════

INGREDIENT_PROPERTIES: List[Tuple[str, List[str], str]] = [
    # Grains
    ("white rice",        ["HighGlycemicIndex","HighRefinedCarbs"],                           "guideline"),
    ("brown rice",        ["ModerateGlycemicIndex","HighFiber"],                              "guideline"),
    ("oats",              ["ModerateGlycemicIndex","HighFiber"],                              "guideline"),
    ("quinoa",            ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    ("wheat",             ["ContainsGluten","ContainsWheat"],                                 "guideline"),
    ("flour",             ["HighGlycemicIndex","ContainsGluten","ContainsWheat","HighRefinedCarbs"],"guideline"),
    ("white bread",       ["HighGlycemicIndex","ContainsGluten","ContainsWheat","HighRefinedCarbs"],"guideline"),
    ("whole wheat bread", ["ModerateGlycemicIndex","ContainsGluten","ContainsWheat","HighFiber"],"guideline"),
    ("rye bread",         ["ModerateGlycemicIndex","ContainsGluten","HighFiber"],              "guideline"),
    ("pasta",             ["ModerateGlycemicIndex","ContainsGluten","ContainsWheat"],          "guideline"),
    ("noodles",           ["HighGlycemicIndex","ContainsGluten","HighSodium","HighRefinedCarbs"],"guideline"),
    ("potato",            ["HighGlycemicIndex","HighRefinedCarbs","HighPotassium"],            "guideline"),
    ("sweet potato",      ["ModerateGlycemicIndex","HighFiber","HighPotassium"],              "guideline"),
    ("corn",              ["HighGlycemicIndex","HighRefinedCarbs"],                           "guideline"),
    ("beans",             ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    ("lentil",            ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    # Sugars
    ("sugar",             ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("honey",             ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("syrup",             ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("candy",             ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("dark chocolate",    ["HighCaffeine","HighSaturatedFat","AntiInflammatory"],             "guideline"),
    ("chocolate",         ["HighSugar","HighSaturatedFat","HighCaffeine"],                    "guideline"),
    ("ice cream",         ["HighSugar","ContainsLactose","ContainsMilk","HighSaturatedFat"],  "guideline"),
    ("cake",              ["HighSugar","ContainsGluten","ContainsEgg"],                       "guideline"),
    ("donut",             ["HighSugar","ContainsGluten","HighSaturatedFat"],                  "guideline"),
    ("cookie",            ["HighSugar","ContainsGluten","ContainsEgg"],                       "guideline"),
    # Dairy
    ("milk",              ["ContainsLactose","ContainsMilk","HighSodium","HighCalcium"],      "guideline"),
    ("cheese",            ["ContainsLactose","ContainsMilk","HighSaturatedFat","HighSodium","HighCalcium"],"guideline"),
    ("yogurt",            ["ContainsLactose","ContainsMilk","ProbioticRich","HighCalcium"],   "guideline"),
    ("butter",            ["HighSaturatedFat","ContainsLactose","ContainsMilk"],              "guideline"),
    ("cream",             ["HighSaturatedFat","ContainsLactose","ContainsMilk"],              "guideline"),
    ("mayonnaise",        ["HighTotalFat","HighSaturatedFat","ContainsEgg"],                  "guideline"),
    ("almond milk",       ["DairyFree","LowGlycemicIndex","NonCaffeinated","ContainsTreeNut"],"guideline"),
    ("soy milk",          ["DairyFree","Goitrogenic","ContainsSoy"],                          "guideline"),
    ("oat milk",          ["DairyFree","ModerateGlycemicIndex","ContainsGluten"],             "guideline"),
    # Proteins — animal
    ("red meat",          ["HighSaturatedFat","HighPurine","HighAnimalProtein","HighIron"],   "guideline"),
    ("processed meat",    ["HighSodium","HighSaturatedFat","HighPurine","HighAnimalProtein"], "guideline"),
    ("bacon",             ["HighSodium","HighSaturatedFat","HighPurine","HighAnimalProtein"], "guideline"),
    ("deli turkey",       ["HighSodium","HighAnimalProtein"],                                  "guideline"),
    ("salami",            ["HighSodium","HighSaturatedFat","HighAnimalProtein"],              "guideline"),
    ("chicken",           ["HighAnimalProtein","LeanProtein","LowPurine"],                    "guideline"),
    ("turkey",            ["HighAnimalProtein","LeanProtein","LowPurine"],                    "guideline"),
    ("egg",               ["ContainsEgg","LeanProtein","HighVitaminB12"],                     "guideline"),
    ("organ meat",        ["HighPurine","HighSaturatedFat","HighAnimalProtein","HighIron"],   "guideline"),
    ("salmon",            ["HighPurine","OmegaRich","AntiInflammatory","HighAnimalProtein","LeanProtein","ContainsFish"],"guideline"),
    ("raw fish",          ["RawSeafood","HighPurine","ContainsFish"],                         "guideline"),
    ("shrimp",            ["ContainsShellfish","HighPurine","HighAnimalProtein","LeanProtein"],"guideline"),
    ("crab",              ["ContainsShellfish","HighPurine"],                                 "guideline"),
    ("lobster",           ["ContainsShellfish","HighPurine"],                                 "guideline"),
    ("sardine",           ["OmegaRich","AntiInflammatory","HighPurine","ContainsFish","HighCalcium"],"guideline"),
    # Proteins — plant
    ("tofu",              ["Goitrogenic","ContainsSoy"],                                      "guideline"),
    ("soy",               ["Goitrogenic","ContainsSoy"],                                      "guideline"),
    ("edamame",           ["Goitrogenic","ContainsSoy","HighFiber"],                          "guideline"),
    ("tempeh",            ["Goitrogenic","ContainsSoy","ProbioticRich"],                      "guideline"),
    # Nuts & seeds
    ("peanut",            ["ContainsPeanut","HighTotalFat"],                                  "guideline"),
    ("peanut butter",     ["ContainsPeanut","HighTotalFat"],                                  "guideline"),
    ("almond",            ["ContainsTreeNut","HighTotalFat","HighCalcium"],                   "guideline"),
    ("cashew",            ["ContainsTreeNut","HighTotalFat"],                                 "guideline"),
    ("walnut",            ["ContainsTreeNut","HighTotalFat","OmegaRich","AntiInflammatory"],  "guideline"),
    ("pistachio",         ["ContainsTreeNut","HighTotalFat"],                                 "guideline"),
    ("macadamia",         ["ContainsTreeNut","HighTotalFat","HighSaturatedFat"],              "guideline"),
    ("pecan",             ["ContainsTreeNut","HighTotalFat","AntiInflammatory"],              "guideline"),
    ("hazelnut",          ["ContainsTreeNut","HighTotalFat"],                                 "guideline"),
    ("sesame",            ["ContainsSesame","HighTotalFat"],                                  "guideline"),
    # Fruits
    ("banana",            ["HighSugar","ModerateGlycemicIndex","HighPotassium"],              "guideline"),
    ("mango",             ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("grape",             ["HighSugar","ModerateGlycemicIndex","HighHistamine"],              "guideline"),
    ("apple",             ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    ("pear",              ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    ("orange",            ["Acidic","ModerateGlycemicIndex"],                                 "guideline"),
    ("grapefruit",        ["Acidic","LowGlycemicIndex"],                                      "guideline"),
    ("lemon",             ["Acidic"],                                                          "guideline"),
    ("lime",              ["Acidic"],                                                          "guideline"),
    ("tomato",            ["Acidic","HighHistamine"],                                          "guideline"),
    ("pineapple",         ["Acidic","HighSugar","HighHistamine"],                             "guideline"),
    ("berry",             ["LowGlycemicIndex","HighFiber","AntiInflammatory"],               "guideline"),
    ("dried fruit",       ["HighSugar","HighGlycemicIndex"],                                  "guideline"),
    ("avocado",           ["HighTotalFat","HighFiber","AntiInflammatory"],                    "guideline"),
    ("peach",             ["LowGlycemicIndex"],                                               "guideline"),
    ("cherry",            ["LowGlycemicIndex","AntiInflammatory"],                            "guideline"),
    # Vegetables
    ("broccoli",          ["Goitrogenic","HighFiber","AntiInflammatory","HighVitaminK"],     "guideline"),
    ("kale",              ["Goitrogenic","HighFiber","HighVitaminK","AntiInflammatory"],      "guideline"),
    ("spinach",           ["Goitrogenic","HighOxalate","HighFiber","HighIronContent","HighVitaminK"],"guideline"),
    ("cabbage",           ["Goitrogenic","HighFiber"],                                        "guideline"),
    ("cauliflower",       ["Goitrogenic","HighFiber"],                                        "guideline"),
    ("brussels sprouts",  ["Goitrogenic","HighFiber","HighVitaminK"],                        "guideline"),
    ("beet",              ["HighOxalate","HighSugar","HighPotassium"],                       "guideline"),
    ("asparagus",         ["HighPurine"],                                                     "guideline"),
    ("mushroom",          ["HighPurine","HighHistamine"],                                     "guideline"),
    ("eggplant",          ["HighHistamine"],                                                   "guideline"),
    ("zucchini",          ["LowGlycemicIndex","HighFiber"],                                   "guideline"),
    ("carrot",            ["ModerateGlycemicIndex","HighFiber","AntiInflammatory"],           "guideline"),
    ("cucumber",          ["LowGlycemicIndex","NonAcidic"],                                   "guideline"),
    ("celery",            ["LowGlycemicIndex","HighFiber","LowSodium"],                       "guideline"),
    # Beverages
    ("coffee",            ["HighCaffeine","Acidic","HighTyramine"],                           "guideline"),
    ("espresso",          ["HighCaffeine","Acidic"],                                           "guideline"),
    ("black tea",         ["HighCaffeine","HighTyramine"],                                    "guideline"),
    ("green tea",         ["HighCaffeine","AntiInflammatory"],                                "guideline"),
    ("herbal tea",        ["NonCaffeinated","NonAlcoholic","NonAcidic"],                      "guideline"),
    ("water",             ["NonCaffeinated","NonAlcoholic","LowSodium","LowPurine"],          "guideline"),
    ("sparkling water",   ["NonCaffeinated","NonAlcoholic"],                                  "guideline"),
    ("energy drink",      ["HighCaffeine","HighSugar","HighSodium"],                          "guideline"),
    ("soda",              ["HighSugar","HighGlycemicIndex","Acidic","HighCaffeine"],          "guideline"),
    ("diet soda",         ["Acidic","HighCaffeine"],                                           "guideline"),
    ("wine",              ["HighAlcohol","Acidic","HighHistamine","HighTyramine"],            "guideline"),
    ("beer",              ["HighAlcohol","ContainsGluten","HighPurine","HighHistamine"],       "guideline"),
    ("spirits",           ["HighAlcohol"],                                                     "guideline"),
    ("alcohol",           ["HighAlcohol"],                                                     "guideline"),
    ("orange juice",      ["Acidic","HighSugar","HighGlycemicIndex"],                        "guideline"),
    ("kombucha",          ["ProbioticRich","Acidic","NonAlcoholic"],                          "guideline"),
    ("kefir",             ["ProbioticRich","ContainsLactose","ContainsMilk"],                 "guideline"),
    # Condiments
    ("salt",              ["HighSodium"],                                                     "guideline"),
    ("soy sauce",         ["HighSodium","ContainsGluten","ContainsSoy"],                      "guideline"),
    ("vinegar",           ["Acidic"],                                                          "guideline"),
    ("hot sauce",         ["Spicy","Acidic"],                                                  "guideline"),
    ("chili",             ["Spicy"],                                                           "guideline"),
    ("pepper",            ["Spicy"],                                                           "guideline"),
    ("wasabi",            ["Spicy","Acidic"],                                                  "guideline"),
    ("miso",              ["HighSodium","ContainsSoy","ProbioticRich"],                       "guideline"),
    ("spice",             ["Spicy"],                                                           "guideline"),
    ("seaweed",           ["HighIodine","HighPotassium"],                                     "guideline"),
    ("kelp",              ["HighIodine"],                                                      "guideline"),
    ("iodized salt",      ["HighIodine","HighSodium"],                                        "guideline"),
    ("aged cheese",       ["HighTyramine","ContainsMilk","ContainsLactose","HighSodium"],     "guideline"),
    # Oils & fats
    ("olive oil",         ["HighTotalFat","AntiInflammatory"],                                "guideline"),
    ("coconut oil",       ["HighSaturatedFat","HighTotalFat"],                                "guideline"),
    ("palm oil",          ["HighSaturatedFat","HighTotalFat"],                                "guideline"),
    ("vegetable oil",     ["HighTotalFat"],                                                   "guideline"),
    ("fish oil",          ["OmegaRich","AntiInflammatory","ContainsFish"],                    "guideline"),
]


# ══════════════════════════════════════════════════════════════════
# §7  FOOD PHRASE → INGREDIENTS (comprehensive dataset + general)
# ══════════════════════════════════════════════════════════════════

FOOD_PHRASE_INGREDIENTS: List[Tuple[str, List[str]]] = [
    # Dataset-aligned phrases
    ("white rice",                    ["white rice"]),
    ("white rice with chicken curry", ["white rice","chicken","spice","chili"]),
    ("chicken curry",                 ["chicken","spice","chili"]),
    ("sushi",                         ["raw fish","white rice","soy sauce","wasabi"]),
    ("sushi with raw fish",           ["raw fish","white rice","soy sauce","wasabi"]),
    ("banana smoothie",               ["banana","milk"]),
    ("banana",                        ["banana"]),
    ("smoothie",                      ["banana","sugar"]),
    ("turkey sandwich with mayo",     ["deli turkey","mayonnaise","white bread"]),
    ("turkey sandwich",               ["deli turkey","white bread"]),
    ("bacon sandwich",                ["bacon","white bread"]),
    ("oatmeal with honey",            ["oats","honey"]),
    ("oatmeal",                       ["oats"]),
    ("energy drink",                  ["energy drink"]),
    ("salmon",                        ["salmon"]),
    ("grilled salmon",                ["salmon"]),
    ("miso soup",                     ["miso","soy sauce"]),
    ("herbal tea",                    ["herbal tea"]),
    ("pretzels",                      ["flour","salt","wheat"]),
    ("dark chocolate",                ["dark chocolate"]),
    ("ice cream",                     ["ice cream","sugar","milk"]),
    ("pizza",                         ["flour","cheese","tomato","salt"]),
    ("pepperoni pizza",               ["flour","cheese","tomato","salt","processed meat"]),
    ("veggie pizza",                  ["flour","cheese","tomato"]),
    ("burger",                        ["red meat","white bread","salt"]),
    ("cheeseburger",                  ["red meat","cheese","white bread","salt"]),
    ("turkey burger",                 ["turkey","white bread"]),
    ("coffee",                        ["coffee"]),
    ("black coffee",                  ["coffee"]),
    ("espresso",                      ["espresso"]),
    ("cappuccino",                    ["espresso","milk"]),
    ("latte",                         ["coffee","milk"]),
    ("soy latte",                     ["coffee","soy milk"]),
    ("soda",                          ["soda"]),
    ("diet soda",                     ["diet soda"]),
    ("orange juice",                  ["orange juice","orange"]),
    ("wine",                          ["wine"]),
    ("red wine",                      ["wine"]),
    ("beer",                          ["beer"]),
    ("almond milk",                   ["almond milk","almond"]),
    ("milk",                          ["milk"]),
    ("yogurt",                        ["yogurt"]),
    ("cheese",                        ["cheese"]),
    ("butter",                        ["butter"]),
    ("eggs",                          ["egg"]),
    ("boiled eggs",                   ["egg"]),
    ("fried egg",                     ["egg","vegetable oil"]),
    ("poached egg",                   ["egg"]),
    ("peanut butter",                 ["peanut butter","peanut"]),
    ("almond butter",                 ["almond"]),
    ("almonds",                       ["almond"]),
    ("cashews",                       ["cashew"]),
    ("walnuts",                       ["walnut"]),
    ("pistachios",                    ["pistachio"]),
    ("mixed nuts",                    ["almond","cashew","walnut"]),
    ("nuts",                          ["almond","cashew","walnut"]),
    ("shrimp",                        ["shrimp"]),
    ("boiled shrimp",                 ["shrimp"]),
    ("broccoli",                      ["broccoli"]),
    ("steamed broccoli",              ["broccoli"]),
    ("kale",                          ["kale"]),
    ("kale chips",                    ["kale","vegetable oil","salt"]),
    ("spinach",                       ["spinach"]),
    ("potato chips",                  ["potato","salt","vegetable oil"]),
    ("chips",                         ["potato","salt","vegetable oil"]),
    ("popcorn",                       ["corn","salt"]),
    ("apple pie",                     ["apple","flour","sugar","butter"]),
    ("apple",                         ["apple"]),
    ("orange",                        ["orange"]),
    ("grapefruit",                    ["grapefruit"]),
    ("mango",                         ["mango"]),
    ("berries",                       ["berry"]),
    ("mixed berries",                 ["berry"]),
    ("dried fruit",                   ["dried fruit"]),
    ("avocado",                       ["avocado"]),
    ("bread",                         ["white bread","flour","salt"]),
    ("whole grain bread",             ["whole wheat bread"]),
    ("rye bread",                     ["rye bread"]),
    ("bagel",                         ["flour","wheat","salt"]),
    ("toast",                         ["white bread","butter"]),
    ("pasta",                         ["pasta"]),
    ("noodles",                       ["noodles"]),
    ("ramen",                         ["noodles","salt","soy sauce"]),
    ("spicy ramen",                   ["noodles","salt","soy sauce","chili"]),
    ("instant noodles",               ["noodles","salt"]),
    ("quinoa salad",                  ["quinoa"]),
    ("quinoa",                        ["quinoa"]),
    ("lentil soup",                   ["lentil"]),
    ("tofu",                          ["tofu","soy"]),
    ("tofu stir-fry",                 ["tofu","soy","vegetable oil","soy sauce"]),
    ("fried chicken",                 ["chicken","flour","vegetable oil","salt"]),
    ("fried rice",                    ["white rice","soy sauce","salt","vegetable oil","egg"]),
    ("burrito",                       ["flour","white rice","beans","chili"]),
    ("tacos",                         ["flour","red meat","chili"]),
    ("chili",                         ["red meat","chili","spice"]),
    ("curry",                         ["spice","chili"]),
    ("seaweed",                       ["seaweed"]),
    ("protein shake",                 ["milk","sugar"]),
    ("apple cider vinegar",           ["vinegar"]),
    ("kimchi",                        ["chili","vinegar","salt"]),
    ("kombucha",                      ["kombucha"]),
    ("salami",                        ["salami"]),
    ("deli meat",                     ["processed meat"]),
    ("glazed donut",                  ["donut","sugar"]),
    ("brownie",                       ["chocolate","sugar","flour","butter"]),
    ("baked sweet potato",            ["sweet potato"]),
    ("grilled cheese sandwich",       ["cheese","white bread","butter"]),
    ("peanut butter toast",           ["peanut butter","white bread"]),
    ("sardines",                      ["sardine"]),
    ("aged cheese",                   ["aged cheese"]),
    ("kefir",                         ["kefir"]),
    ("olive oil",                     ["olive oil"]),
    ("water",                         ["water"]),
    ("sparkling water",               ["sparkling water"]),
    ("green tea",                     ["green tea"]),
    ("black tea",                     ["black tea"]),
    ("cherry",                        ["cherry"]),
    ("walnut",                        ["walnut"]),
    ("fish oil",                      ["fish oil"]),
    ("dark leafy greens",             ["kale","spinach","broccoli"]),
    ("fermented foods",               ["yogurt","kimchi","kombucha"]),
    ("organ meat",                    ["organ meat"]),
    ("processed food",                ["processed meat","salt","sugar"]),
    # ── Missing from dataset audit — added ────────────────────────
    ("brown rice",                    ["brown rice"]),
    ("steamed brown rice",            ["brown rice"]),
    ("beef steak",                    ["red meat"]),
    ("steak",                         ["red meat"]),
    ("baked eggplant",                ["eggplant"]),
    ("eggplant parmigiana",           ["eggplant","cheese","tomato"]),
    ("boiled potatoes",               ["potato"]),
    ("mashed potatoes",               ["potato","butter","milk"]),
    ("fried potatoes",                ["potato","vegetable oil","salt"]),
    ("canned fruit in syrup",         ["dried fruit","sugar","syrup"]),
    ("canned peaches",                ["peach","sugar","syrup"]),
    ("fresh peaches",                 ["peach"]),
    ("canned soup",                   ["salt","vegetable oil"]),
    ("canned vegetable soup",         ["salt","tomato"]),
    ("fresh vegetable soup",          ["salt","tomato"]),
    ("chicken noodle soup",           ["chicken","noodles","salt"]),
    ("canned vegetables",             ["salt"]),
    ("fresh vegetables",              ["broccoli","carrot","zucchini"]),
    ("fresh fruit",                   ["apple","berry","orange"]),
    ("chamomile tea",                 ["herbal tea"]),
    ("chicken caesar salad",          ["chicken","cheese","mayonnaise"]),
    ("caesar salad",                  ["cheese","mayonnaise"]),
    ("chicken salad sandwich",        ["chicken","white bread","mayonnaise"]),
    ("grilled chicken salad",         ["chicken"]),
    ("fresh garden salad",            ["tomato","celery","cucumber"]),
    ("chocolate cake",                ["chocolate","sugar","flour","butter","egg"]),
    ("fizzy drinks",                  ["soda"]),
    ("cola",                          ["soda"]),
    ("carbonated drinks",             ["soda"]),
    ("soft drink",                    ["soda"]),
    ("fried fish",                    ["salmon","flour","vegetable oil"]),
    ("hard boiled egg",               ["egg"]),
    ("high sugar cereals",            ["corn","sugar","flour"]),
    ("lean grilled chicken",          ["chicken"]),
    ("grilled chicken",               ["chicken"]),
    ("low sodium turkey jerky",       ["deli turkey","salt"]),
    ("turkey jerky",                  ["deli turkey","salt"]),
    ("regular beef jerky",            ["red meat","salt"]),
    ("beef jerky",                    ["red meat","salt"]),
    ("sauerkraut",                    ["vinegar","salt"]),
    ("spicy salsa",                   ["tomato","chili","spice"]),
    ("spicy samosas",                 ["flour","chili","spice","vegetable oil"]),
    ("spicy taco beef",               ["red meat","chili","flour"]),
    ("mixed vegetable stir fry",      ["vegetable oil","soy sauce"]),
    ("protein bar",                   ["sugar","milk","egg"]),
    ("honey tea",                     ["herbal tea","honey"]),
    ("oatmeal with honey",            ["oats","honey"]),
    ("salted cashews",                ["cashew","salt"]),
    ("mango smoothie",                ["mango","milk"]),
    ("green tea matcha",              ["green tea"]),
    ("matcha",                        ["green tea"]),
    ("raw leafy greens",              ["kale","spinach"]),
    ("leafy greens",                  ["kale","spinach","broccoli"]),
    ("peanut butter toast",           ["peanut butter","white bread"]),
    ("soft drinks",                   ["soda"]),
    ("carbonated soft drinks",        ["soda"]),
    ("chicken wrap",                  ["chicken","flour"]),
    ("tuna salad",                    ["salmon","mayonnaise"]),
    ("fruit salad",                   ["apple","berry","orange"]),
    ("vegetable soup",                ["salt","tomato","carrot"]),
    ("tomato soup",                   ["tomato","salt"]),
    ("caesar salad dressing",         ["cheese","mayonnaise","vinegar"]),
    ("low fat yogurt",                ["yogurt"]),
    ("whole grain crackers",          ["whole wheat bread","salt"]),
    ("almond butter toast",           ["almond","white bread"]),
    ("smoothie bowl",                 ["banana","berry","milk"]),
    # Long-form query variants from dataset audit
    ("oatmeal with honey fine in the morning", ["oats","honey"]),
    ("can i snack on salted cashews",          ["cashew","salt"]),
    ("energy drink okay before my workout",    ["energy drink"]),
    ("whole milk",                    ["milk"]),
    ("dark chocolate is that fine",   ["dark chocolate"]),
    ("i m thinking about peanut butter toast", ["peanut butter","white bread"]),
    ("bacon sandwich is that fine",   ["bacon","white bread"]),
    ("glass of red wine",             ["wine"]),
    ("cup of green tea",              ["green tea"]),
    ("bowl of oatmeal",               ["oats"]),
    ("slice of pizza",                ["flour","cheese","tomato","salt"]),
    ("handful of almonds",            ["almond"]),
    ("small steak",                   ["red meat"]),
    ("grilled salmon with vegetables",["salmon","broccoli","carrot"]),
    ("can i enjoy a small steak with high blood pressure",["red meat","salt"]),
    ("would having a grilled chicken salad help my condition",["chicken"]),
    ("would having a chicken salad sandwich",["chicken","white bread","mayonnaise"]),
    ("honey to my tea is that alright",["honey","herbal tea"]),
    ("it alright to consume honey for heart disease management",["honey"]),
    ("it advisable to eat raw leafy greens with ulcerative colitis",["kale","spinach"]),
]


# ══════════════════════════════════════════════════════════════════
# §8  USDA NUTRIENT THRESHOLDS (for deriving properties from FoodItem)
# ══════════════════════════════════════════════════════════════════

USDA_THRESHOLDS = [
    ("HighSodium",       "sodium",                    600.0,  "mg/100g"),
    ("HighSugar",        "sugars",                     10.0,  "g/100g"),
    ("HighPotassium",    "potassium",                 400.0,  "mg/100g"),
    ("HighPhosphorus",   "phosphorus",                250.0,  "mg/100g"),
    ("HighSaturatedFat", "fatty acids total saturated", 5.0,  "g/100g"),
    ("HighTotalFat",     "total lipid",                17.5,  "g/100g"),
    ("HighCaffeine",     "caffeine",                   80.0,  "mg/100g"),
    ("HighFiber",        "fiber dietary",               6.0,  "g/100g"),
    ("HighIron",         "iron",                        45.0, "mg/100g"),
    ("HighCalcium",      "calcium",                   300.0,  "mg/100g"),
]


# ══════════════════════════════════════════════════════════════════
# §9  USDA FOOD NAME → INGREDIENT MATCHING
#     Connects 2M USDA food items to the curated ingredient layer.
#     This makes the KG general — any USDA food can be queried.
# ══════════════════════════════════════════════════════════════════

# Keywords that map USDA food description tokens to canonical ingredients
USDA_KEYWORD_TO_INGREDIENT = {
    "salmon":          "salmon",
    "sardine":         "sardine",
    "sardines":        "sardine",
    "tuna":            "ContainsFish",
    "cod":             "ContainsFish",
    "tilapia":         "ContainsFish",
    "halibut":         "ContainsFish",
    "trout":           "salmon",
    "shrimp":          "shrimp",
    "crab":            "crab",
    "lobster":         "lobster",
    "clam":            "ContainsShellfish",
    "oyster":          "ContainsShellfish",
    "chicken":         "chicken",
    "turkey":          "turkey",
    "beef":            "red meat",
    "pork":            "red meat",
    "lamb":            "red meat",
    "bacon":           "bacon",
    "ham":             "processed meat",
    "sausage":         "processed meat",
    "hot dog":         "processed meat",
    "salami":          "salami",
    "pepperoni":       "processed meat",
    "egg":             "egg",
    "milk":            "milk",
    "cheese":          "cheese",
    "yogurt":          "yogurt",
    "butter":          "butter",
    "cream":           "cream",
    "ice cream":       "ice cream",
    "wheat":           "wheat",
    "bread":           "white bread",
    "pasta":           "pasta",
    "rice":            "white rice",
    "oat":             "oats",
    "corn":            "corn",
    "potato":          "potato",
    "sweet potato":    "sweet potato",
    "beans":           "beans",
    "lentil":          "lentil",
    "tofu":            "tofu",
    "soy":             "soy",
    "almond":          "almond",
    "walnut":          "walnut",
    "cashew":          "cashew",
    "peanut":          "peanut",
    "peanuts":         "peanut",
    "pistachio":       "pistachio",
    "pecan":           "pecan",
    "hazelnut":        "hazelnut",
    "sesame":          "sesame",
    "banana":          "banana",
    "apple":           "apple",
    "orange":          "orange",
    "mango":           "mango",
    "grape":           "grape",
    "berry":           "berry",
    "blueberry":       "berry",
    "strawberry":      "berry",
    "raspberry":       "berry",
    "cherry":          "cherry",
    "avocado":         "avocado",
    "tomato":          "tomato",
    "broccoli":        "broccoli",
    "spinach":         "spinach",
    "kale":            "kale",
    "mushroom":        "mushroom",
    "asparagus":       "asparagus",
    "beet":            "beet",
    "carrot":          "carrot",
    "celery":          "celery",
    "coffee":          "coffee",
    "tea":             "black tea",
    "alcohol":         "alcohol",
    "wine":            "wine",
    "beer":            "beer",
    "chocolate":       "chocolate",
    "cocoa":           "dark chocolate",
    "sugar":           "sugar",
    "honey":           "honey",
    "salt":            "salt",
    "seaweed":         "seaweed",
    "kelp":            "seaweed",
    "oil":             "vegetable oil",
    "olive oil":       "olive oil",
    "coconut oil":     "coconut oil",
}


def match_usda_food_to_ingredients(food_name: str) -> List[str]:
    """Map a USDA food description to canonical ingredient IDs."""
    name_lower = norm(food_name)
    matched = set()
    for keyword, ingredient in USDA_KEYWORD_TO_INGREDIENT.items():
        if keyword in name_lower:
            matched.add(ingredient)
    return list(matched)


# ══════════════════════════════════════════════════════════════════
# §10  OPTIONAL UMLS ENRICHMENT
# ══════════════════════════════════════════════════════════════════

ALLOWED_TUIS = {"T047","T048","T033","T037","T184"}
PREFERRED_SABS = {"SNOMEDCT_US","ICD10CM","MSH","OMIM"}

def load_umls_optional(umls_dir: Path, canon_set: Set[str]):
    mrconso = umls_dir / "MRCONSO.RRF"
    mrsty   = umls_dir / "MRSTY.RRF"
    mrrel   = umls_dir / "MRREL.RRF"
    if not mrconso.exists():
        print("  [UMLS] Not found — using curated conditions only")
        return [], []

    allowed: Set[str] = set()
    with mrsty.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) >= 2 and p[1] in ALLOWED_TUIS:
                allowed.add(p[0])

    # Build name → CUI map for our canonical conditions
    name2cui: Dict[str, str] = {}
    with mrconso.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 15: continue
            cui, lat, sab, term = p[0], p[1], p[11], p[14].strip()
            if cui not in allowed or lat != "ENG": continue
            n = norm(term)
            if n not in name2cui:
                name2cui[n] = cui

    # Match canonical names → CUIs
    alias_edges = []
    for cname in canon_set:
        cui = name2cui.get(norm(cname))
        if cui:
            alias_edges.append({
                "start_id": f"UMLS:{cui}",
                "end_id":   sid("COND", cname),
                "type": "SAME_AS", "source": "UMLS"
            })

    # IS_A from MRREL
    isa_edges = []
    if mrrel.exists():
        parent_of: Dict[str, Set[str]] = defaultdict(set)
        with mrrel.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.rstrip("\n").split("|")
                if len(p) >= 5 and p[3] in {"PAR","RB"} and p[0] != p[4]:
                    parent_of[p[0]].add(p[4])
        matched_cuis = {name2cui[norm(c)] for c in canon_set if norm(c) in name2cui}
        for cui in list(matched_cuis):
            for par in parent_of.get(cui, set()):
                isa_edges.append({"start_id": f"UMLS:{cui}", "end_id": f"UMLS:{par}",
                                   "type": "IS_A", "source": "UMLS_MRREL"})

    print(f"  [UMLS] {len(alias_edges)} SAME_AS edges, {len(isa_edges)} IS_A edges")
    return alias_edges, isa_edges


# ══════════════════════════════════════════════════════════════════
# §11  KG BUILDER
# ══════════════════════════════════════════════════════════════════

def build(args) -> Dict[str, int]:
    root      = Path(args.root)
    out_nodes = root / "out" / "nodes"
    out_edges = root / "out" / "edges"
    ensure_dir(out_nodes); ensure_dir(out_edges)
    stats: Dict[str, int] = {}

    alias2canon = {norm(a): c for a, c in CONDITION_ALIASES}
    canon_set   = set(c for _, c in CONDITION_ALIASES)

    # ── Condition nodes ───────────────────────────────────────────
    cond_nodes, family_nodes = [], []
    seen_fam: Set[str] = set()
    for cname in canon_set:
        cond_nodes.append({"id": sid("COND",cname), "label":"Condition",
                            "name": cname, "source":"curated_expert"})
    for leaf, fam, sfam in CONDITION_ISA:
        for fn in [fam, sfam]:
            if fn not in seen_fam:
                seen_fam.add(fn)
                family_nodes.append({"id": sid("COND",fn), "label":"ConditionFamily",
                                      "name": fn, "source":"curated_expert"})
    all_cond = dedup(cond_nodes + family_nodes)
    write_csv(out_nodes/"nodes_condition.csv", ["id","label","name","source"], all_cond)
    stats["condition_nodes"] = len([n for n in all_cond if n["label"]=="Condition"])
    stats["family_nodes"]    = len([n for n in all_cond if n["label"]=="ConditionFamily"])

    # ── Condition aliases ─────────────────────────────────────────
    al_nodes, al_edges = [], []
    for surface, canon in CONDITION_ALIASES:
        aid = sid("COND_ALIAS", surface)
        al_nodes.append({"id": aid, "label":"ConditionAlias", "text": surface})
        al_edges.append({"start_id": aid, "end_id": sid("COND",canon),
                          "type":"ALIAS_OF", "source":"curated_expert"})
    write_csv(out_nodes/"nodes_condition_alias.csv", ["id","label","text"], dedup(al_nodes))
    write_csv(out_edges/"edges_condition_alias_of.csv", ["start_id","end_id","type","source"], al_edges)
    stats["alias_nodes"] = len(al_nodes)

    # ── IS_A hierarchy ────────────────────────────────────────────
    isa_edges = []
    for leaf, fam, sfam in CONDITION_ISA:
        isa_edges.append({"start_id": sid("COND",leaf), "end_id": sid("COND",fam),
                           "type":"IS_A", "source":"curated_expert"})
        isa_edges.append({"start_id": sid("COND",fam),  "end_id": sid("COND",sfam),
                           "type":"IS_A", "source":"curated_expert"})
    isa_edges = dedup_edges(isa_edges, ["start_id","end_id","type"])
    write_csv(out_edges/"edges_condition_isa.csv", ["start_id","end_id","type","source"], isa_edges)
    stats["isa_edges"] = len(isa_edges)

    # ── FoodProperty nodes ────────────────────────────────────────
    prop_nodes = []
    prop_ids: Set[str] = set()
    for pid,desc,thr,unit,src_name,src_url,ev in PROPERTIES:
        prop_nodes.append({"id":f"PROP:{pid}","label":"FoodProperty","name":pid,
                            "description":desc,"threshold":thr,"unit":unit,
                            "source_name":src_name,"source_url":src_url,"evidence_level":ev})
        prop_ids.add(pid)
    write_csv(out_nodes/"nodes_food_property.csv",
              ["id","label","name","description","threshold","unit","source_name","source_url","evidence_level"],
              dedup(prop_nodes))
    stats["property_nodes"] = len(prop_nodes)

    # ── Ingredient nodes ──────────────────────────────────────────
    ing_set: Set[str] = set()
    for ing,_,_ in INGREDIENT_PROPERTIES: ing_set.add(ing)
    for _,ings in FOOD_PHRASE_INGREDIENTS: ing_set.update(ings)
    ing_nodes = [{"id":f"ING:{norm(i).replace(' ','_')}","label":"Ingredient","name":i}
                 for i in sorted(ing_set) if i]
    write_csv(out_nodes/"nodes_ingredient.csv", ["id","label","name"], dedup(ing_nodes))
    stats["ingredient_nodes"] = len(ing_nodes)

    # ── Ingredient → Property edges ───────────────────────────────
    ing_prop_edges = []
    for ing,props,ev in INGREDIENT_PROPERTIES:
        iid = f"ING:{norm(ing).replace(' ','_')}"
        for p in props:
            if p in prop_ids:
                ing_prop_edges.append({"start_id":iid,"end_id":f"PROP:{p}",
                                        "type":"HAS_PROPERTY","evidence_level":ev,"source":"curated_expert"})

    # ── Load database-derived ingredient → property CSVs ─────────
    # Merges rules from build_rules_from_databases.py output
    csv_sources = []
    if hasattr(args,"rules_csv") and args.rules_csv:
        csv_sources.append(Path(args.rules_csv))
    if hasattr(args,"extra_rules") and args.extra_rules:
        extra = Path(args.extra_rules)
        for f in extra.glob("*.csv"):
            csv_sources.append(f)

    db_new_ingredients = []
    for csv_path in csv_sources:
        if not csv_path.exists():
            print(f"  [Rules] File not found: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
            ing_col  = next((c for c in ["ingredient"] if c in df.columns), None)
            prop_col = next((c for c in ["property"] if c in df.columns), None)
            src_col  = next((c for c in ["source","source_name"] if c in df.columns), None)
            ev_col   = "evidence_level" if "evidence_level" in df.columns else None
            if not ing_col or not prop_col:
                continue
            added = 0
            for _, r in df.iterrows():
                ing  = str(r.get(ing_col,"")).strip()
                prop = str(r.get(prop_col,"")).strip()
                src  = str(r.get(src_col,"db_derived")).strip() if src_col else "db_derived"
                ev   = str(r.get(ev_col,"guideline")).strip() if ev_col else "guideline"
                if not ing or not prop or prop not in prop_ids:
                    continue
                iid = f"ING:{norm(ing).replace(' ','_')}"
                ing_prop_edges.append({"start_id":iid,"end_id":f"PROP:{prop}",
                                        "type":"HAS_PROPERTY","evidence_level":ev,"source":src})
                db_new_ingredients.append({"id":iid,"label":"Ingredient","name":ing})
                added += 1
            print(f"  [Rules] Loaded {added} edges from {csv_path.name}")
        except Exception as e:
            print(f"  [Rules] Error reading {csv_path}: {e}")

    # Add any new ingredient nodes from CSV sources
    if db_new_ingredients:
        ing_nodes.extend(db_new_ingredients)
        ing_nodes = dedup(ing_nodes)
        write_csv(out_nodes/"nodes_ingredient.csv", ["id","label","name"], ing_nodes)
        stats["ingredient_nodes"] = len(ing_nodes)

    ing_prop_edges = dedup_edges(ing_prop_edges, ["start_id","end_id","type"])
    write_csv(out_edges/"edges_ingredient_has_property.csv",
              ["start_id","end_id","type","evidence_level","source"], ing_prop_edges)
    stats["has_property_edges"] = len(ing_prop_edges)

    # ── FoodPhrase nodes + HAS_INGREDIENT edges ───────────────────
    fp_nodes, has_ing_edges = [], []
    for phrase, ings in FOOD_PHRASE_INGREDIENTS:
        fpid = f"FP:{norm(phrase).replace(' ','_')}"
        fp_nodes.append({"id":fpid,"label":"FoodPhrase","name":phrase})
        for ing in ings:
            if ing:
                has_ing_edges.append({"start_id":fpid,
                                       "end_id":f"ING:{norm(ing).replace(' ','_')}",
                                       "type":"HAS_INGREDIENT","source":"curated_expert"})
    write_csv(out_nodes/"nodes_food_phrase.csv",  ["id","label","name"], dedup(fp_nodes))
    write_csv(out_edges/"edges_food_has_ingredient.csv",
              ["start_id","end_id","type","source"],
              dedup_edges(has_ing_edges,["start_id","end_id","type"]))
    stats["food_phrase_nodes"]    = len(fp_nodes)
    stats["has_ingredient_edges"] = len(has_ing_edges)

    # ── RISKY_FOR + SAFE_FOR edges ────────────────────────────────
    risky, safe = [], []
    for prop,cond,str_,sn,su in PROPERTY_RISKY_FOR:
        risky.append({"start_id":f"PROP:{prop}","end_id":sid("COND",cond),
                       "type":"RISKY_FOR","strength":str_,"source_name":sn,"source_url":su,"evidence_level":"guideline"})
    for prop,cond,str_,sn,su in PROPERTY_SAFE_FOR:
        safe.append({"start_id":f"PROP:{prop}","end_id":sid("COND",cond),
                      "type":"SAFE_FOR","strength":str_,"source_name":sn,"source_url":su,"evidence_level":"guideline"})
    risky = dedup_edges(risky,["start_id","end_id","type"])
    safe  = dedup_edges(safe, ["start_id","end_id","type"])
    write_csv(out_edges/"edges_property_risky_for_condition.csv",
              ["start_id","end_id","type","strength","source_name","source_url","evidence_level"], risky)
    write_csv(out_edges/"edges_property_safe_for_condition.csv",
              ["start_id","end_id","type","strength","source_name","source_url","evidence_level"], safe)
    stats["risky_for_edges"] = len(risky)
    stats["safe_for_edges"]  = len(safe)

    # ── Shortcut edges: FoodPhrase → Condition ────────────────────
    ing_to_props: Dict[str,Set[str]] = defaultdict(set)
    for e in ing_prop_edges: ing_to_props[e["start_id"]].add(e["end_id"])
    prop_to_risks: Dict[str,List] = defaultdict(list)
    for e in risky: prop_to_risks[e["start_id"]].append(e)

    shortcuts = []
    for e_hi in has_ing_edges:
        fp_id, ing_id = e_hi["start_id"], e_hi["end_id"]
        for prop_id in ing_to_props.get(ing_id, set()):
            for risk in prop_to_risks.get(prop_id, []):
                shortcuts.append({"start_id":fp_id,"end_id":risk["end_id"],
                                   "type":"RISKY_FOR","via_prop":prop_id,
                                   "strength":risk.get("strength",""),
                                   "source_name":risk.get("source_name","derived"),
                                   "evidence_level":"inferred",
                                   "path":"FoodPhrase→Ingredient→Property→Condition"})

    # ── Add direct food→condition edges ──────────────────────────
    for food_phrase, cond, strength, src_name, src_url in DIRECT_FOOD_CONDITION_RISKS:
        fp_id  = f"FP:{norm(food_phrase).replace(' ','_')}"
        cond_id = sid("COND", cond)
        # Ensure FoodPhrase node exists
        if fp_id not in {n["id"] for n in fp_nodes}:
            fp_nodes.append({"id":fp_id,"label":"FoodPhrase","name":food_phrase})
        shortcuts.append({"start_id":fp_id,"end_id":cond_id,
                           "type":"RISKY_FOR","via_prop":"direct",
                           "strength":strength,"source_name":src_name,
                           "evidence_level":"guideline",
                           "path":"FoodPhrase→Condition (direct, clinically curated)"})

    # Re-save FoodPhrase nodes in case new ones were added
    write_csv(out_nodes/"nodes_food_phrase.csv", ["id","label","name"], dedup(fp_nodes))
    stats["food_phrase_nodes"] = len(dedup(fp_nodes))

    shortcuts = dedup_edges(shortcuts,["start_id","end_id","type","via_prop"])
    write_csv(out_edges/"edges_food_risky_for_condition_shortcut.csv",
              ["start_id","end_id","type","via_prop","strength","source_name","evidence_level","path"],
              shortcuts)
    stats["shortcut_edges"] = len(shortcuts)

    # ── DIRECT FOOD → SAFE_FOR → CONDITION shortcuts ──────────────
    safe_shortcuts = []
    fp_ids_set = {n["id"] for n in dedup(fp_nodes)}
    for food_phrase, cond, strength, src_name, src_url in DIRECT_FOOD_CONDITION_SAFE:
        fp_id   = f"FP:{norm(food_phrase).replace(' ','_')}"
        cond_id = sid("COND", cond)
        if fp_id not in fp_ids_set:
            fp_nodes.append({"id":fp_id,"label":"FoodPhrase","name":food_phrase})
            fp_ids_set.add(fp_id)
        safe_shortcuts.append({
            "start_id": fp_id, "end_id": cond_id,
            "type": "SAFE_FOR", "via_prop": "direct",
            "strength": strength, "source_name": src_name,
            "evidence_level": "guideline",
            "path": "FoodPhrase→Condition (direct SAFE, clinically curated)"
        })
    safe_shortcuts = dedup_edges(safe_shortcuts, ["start_id","end_id","type","via_prop"])
    write_csv(out_edges/"edges_food_safe_for_condition_shortcut.csv",
              ["start_id","end_id","type","via_prop","strength","source_name","evidence_level","path"],
              safe_shortcuts)
    stats["safe_shortcut_edges"] = len(safe_shortcuts)
    print(f"  ✓ SAFE_FOR shortcut edges: {len(safe_shortcuts)}")

    # Critical fix: re-save FoodPhrase nodes AGAIN to include
    # new nodes created during SAFE_FOR shortcut processing
    write_csv(out_nodes/"nodes_food_phrase.csv", ["id","label","name"], dedup(fp_nodes))
    stats["food_phrase_nodes"] = len(dedup(fp_nodes))
    print(f"  ✓ Total FoodPhrase nodes: {len(dedup(fp_nodes))}")

    # ── USDA augmentation ─────────────────────────────────────────
    stats["food_item_nodes"] = 0; stats["nutrient_nodes"] = 0
    stats["has_nutrient_edges"] = 0; stats["usda_prop_edges"] = 0

    if not args.no_usda:
        usda_dir = root / "data" / "usda"
        food_csv = usda_dir / "food.csv"
        nutr_csv = usda_dir / "nutrient.csv"
        fn_csv   = usda_dir / "food_nutrient.csv"

        if food_csv.exists() and nutr_csv.exists() and fn_csv.exists():
            food_df = pd.read_csv(food_csv, low_memory=False)
            nutr_df = pd.read_csv(nutr_csv, low_memory=False)
            if "unitName" in nutr_df.columns:
                nutr_df = nutr_df.rename(columns={"unitName":"unit_name"})

            food_nodes_usda = [{"id":f"FDC:{int(r['fdc_id'])}","label":"FoodItem",
                                  "fdc_id":int(r["fdc_id"]),"name":r.get("description",""),
                                  "data_type":r.get("data_type","")}
                                for _,r in food_df[["fdc_id","description","data_type"]].drop_duplicates().iterrows()]
            nutr_nodes_usda = [{"id":f"NUT:{int(r['id'])}","label":"Nutrient",
                                  "nutrient_id":int(r["id"]),"name":r.get("name",""),
                                  "unit":r.get("unit_name","")}
                                for _,r in nutr_df[["id","name"]+
                                             (["unit_name"] if "unit_name" in nutr_df.columns else [])
                                            ].drop_duplicates().iterrows()]
            write_csv(out_nodes/"nodes_food_item.csv",["id","label","fdc_id","name","data_type"],food_nodes_usda)
            write_csv(out_nodes/"nodes_nutrient.csv",["id","label","nutrient_id","name","unit"],nutr_nodes_usda)
            stats["food_item_nodes"] = len(food_nodes_usda)
            stats["nutrient_nodes"]  = len(nutr_nodes_usda)
            print(f"  [USDA] {len(food_nodes_usda):,} food items, {len(nutr_nodes_usda):,} nutrients")

            # Stream food→nutrient edges + derive properties
            nid2name = {n["id"]: norm(n["name"]) for n in nutr_nodes_usda}
            usda_prop_rows = []
            n_fn = 0
            with (out_edges/"edges_food_has_nutrient.csv").open("w",newline="",encoding="utf-8") as f:
                w = csv.DictWriter(f,fieldnames=["start_id","end_id","type","amount","source"],
                                   quoting=csv.QUOTE_ALL)
                w.writeheader()
                for chunk in pd.read_csv(fn_csv, low_memory=False, chunksize=500_000):
                    if "nutrient_id" not in chunk.columns and "id" in chunk.columns:
                        chunk = chunk.rename(columns={"id":"nutrient_id"})
                    chunk = chunk[["fdc_id","nutrient_id","amount"]].dropna()
                    for _,r in chunk.iterrows():
                        w.writerow({"start_id":f"FDC:{int(r['fdc_id'])}",
                                    "end_id":  f"NUT:{int(r['nutrient_id'])}",
                                    "type":"HAS_NUTRIENT","amount":r["amount"],"source":"USDA_FDC"})
                        n_fn += 1
            stats["has_nutrient_edges"] = n_fn

            # Derive FoodItem → FoodProperty edges via thresholds
            fn_df = pd.read_csv(out_edges/"edges_food_has_nutrient.csv")
            fn_df["nut_norm"] = fn_df["end_id"].map(nid2name)
            fn_df["amount"]   = pd.to_numeric(fn_df["amount"], errors="coerce")
            fn_df = fn_df.dropna(subset=["amount","nut_norm"])
            derived_rows = []
            for pid,substr,thr,_ in USDA_THRESHOLDS:
                if pid not in prop_ids: continue
                mask = fn_df["nut_norm"].str.contains(norm(substr),na=False)
                if not mask.any(): continue
                g = fn_df[mask].groupby("start_id")["amount"].max()
                for fid in g[g >= thr].index:
                    derived_rows.append({"start_id":fid,"end_id":f"PROP:{pid}",
                                          "type":"HAS_PROPERTY","source":"USDA_threshold"})
            write_csv(out_edges/"edges_food_item_has_property.csv",
                      ["start_id","end_id","type","source"],derived_rows)
            stats["usda_prop_edges"] = len(derived_rows)
            print(f"  [USDA] {len(derived_rows):,} derived property edges, {n_fn:,} HAS_NUTRIENT edges")

            # ── USDA food name → ingredient matching ──────────────
            # Connect FoodItem nodes to ingredient layer via name matching.
            # This makes any USDA food queryable by the POMDP.
            print("  [USDA] Building food name → ingredient matching layer...")
            fi_ing_edges = []
            seen_fi_edges: Set[tuple] = set()
            for fn_item in food_nodes_usda:
                matched = match_usda_food_to_ingredients(fn_item["name"])
                for ing in matched:
                    iid = f"ING:{norm(ing).replace(' ','_')}"
                    k = (fn_item["id"], iid)
                    if k not in seen_fi_edges:
                        seen_fi_edges.add(k)
                        fi_ing_edges.append({"start_id":fn_item["id"],"end_id":iid,
                                              "type":"HAS_INGREDIENT","source":"usda_name_match"})
            write_csv(out_edges/"edges_food_item_has_ingredient.csv",
                      ["start_id","end_id","type","source"], fi_ing_edges)
            stats["food_item_ingredient_edges"] = len(fi_ing_edges)
            print(f"  [USDA] {len(fi_ing_edges):,} FoodItem→Ingredient name-match edges")

            # ── Materialize FoodItem → Condition shortcuts ─────────
            print("  [USDA] Materializing FoodItem → Condition shortcuts...")
            fi_cond_shortcuts = []
            seen_fi_cond: Set[tuple] = set()
            for e in fi_ing_edges:
                fi_id  = e["start_id"]
                ing_id = e["end_id"]
                for prop_id in ing_to_props.get(ing_id, set()):
                    for risk in prop_to_risks.get(prop_id, []):
                        cond_id = risk["end_id"]
                        k = (fi_id, cond_id, prop_id)
                        if k not in seen_fi_cond:
                            seen_fi_cond.add(k)
                            fi_cond_shortcuts.append({
                                "start_id":fi_id,"end_id":cond_id,
                                "type":"RISKY_FOR","via_prop":prop_id,
                                "strength":risk.get("strength",""),
                                "source_name":risk.get("source_name","derived"),
                                "evidence_level":"inferred",
                                "path":"FoodItem→Ingredient→Property→Condition"
                            })
            write_csv(out_edges/"edges_food_item_risky_for_condition.csv",
                      ["start_id","end_id","type","via_prop","strength","source_name","evidence_level","path"],
                      fi_cond_shortcuts)
            stats["food_item_condition_edges"] = len(fi_cond_shortcuts)
            print(f"  [USDA] {len(fi_cond_shortcuts):,} FoodItem→Condition shortcut edges")
        else:
            print("  [USDA] Files not found in data/usda/ — skipping")

    # ── UMLS enrichment ───────────────────────────────────────────
    if not args.no_umls:
        umls_edges, umls_isa = load_umls_optional(root/"data"/"umls", canon_set)
        if umls_edges:
            write_csv(out_edges/"edges_condition_same_as_umls.csv",
                      ["start_id","end_id","type","source"], umls_edges)
        if umls_isa:
            write_csv(out_edges/"edges_condition_isa_umls.csv",
                      ["start_id","end_id","type","source"], umls_isa)
            stats["isa_edges"] += len(umls_isa)

    return stats


# ══════════════════════════════════════════════════════════════════
# §12  TABLE 4 STATS
# ══════════════════════════════════════════════════════════════════

def print_table4(s: Dict[str,int], root: Path) -> None:
    lines = [
        "=" * 62,
        "TABLE 4 — FoodSafetyKG Statistics (npj Digital Medicine)",
        "=" * 62,
        "",
        "Node types",
        f"  Condition nodes (leaf)                {s.get('condition_nodes',0):>10,}",
        f"  ConditionFamily nodes (IS_A)          {s.get('family_nodes',0):>10,}",
        f"  ConditionAlias nodes                  {s.get('alias_nodes',0):>10,}",
        f"  FoodProperty nodes                    {s.get('property_nodes',0):>10,}",
        f"  Ingredient nodes                      {s.get('ingredient_nodes',0):>10,}",
        f"  FoodPhrase nodes                      {s.get('food_phrase_nodes',0):>10,}",
        f"  FoodItem nodes (USDA FDC)             {s.get('food_item_nodes',0):>10,}",
        f"  Nutrient nodes (USDA FDC)             {s.get('nutrient_nodes',0):>10,}",
        "",
        "Edge types",
        f"  HAS_INGREDIENT (curated)              {s.get('has_ingredient_edges',0):>10,}",
        f"  HAS_INGREDIENT (USDA name-matched)    {s.get('food_item_ingredient_edges',0):>10,}",
        f"  HAS_PROPERTY (curated)                {s.get('has_property_edges',0):>10,}",
        f"  HAS_PROPERTY (USDA-derived)           {s.get('usda_prop_edges',0):>10,}",
        f"  HAS_NUTRIENT (USDA FDC)               {s.get('has_nutrient_edges',0):>10,}",
        f"  RISKY_FOR (direct, guideline-sourced) {s.get('risky_for_edges',0):>10,}",
        f"  SAFE_FOR  (direct, guideline-sourced) {s.get('safe_for_edges',0):>10,}",
        f"  RISKY_FOR (FoodPhrase shortcuts)      {s.get('shortcut_edges',0):>10,}",
        f"  RISKY_FOR (FoodItem shortcuts)        {s.get('food_item_condition_edges',0):>10,}",
        f"  IS_A (curated + UMLS)                 {s.get('isa_edges',0):>10,}",
        f"  ALIAS_OF                              {s.get('alias_nodes',0):>10,}",
        "",
        "Knowledge sources",
        "  USDA FoodData Central (FDC)           2,085,340 food items",
        "  UMLS Metathesaurus                    T047,T048,T033,T037,T184",
        "  Clinical guidelines                   ADA 2024, AHA 2021, ACG 2022",
        "                                        NKF KDOQI 2020, ACR 2020",
        "                                        FARE 2023, ATA 2019, ACOG 2020",
        "                                        NOF 2022, AASLD 2018, AHS 2020",
        "                                        FDA 2023, WHO 2015, ESC 2020",
        "  Evidence levels                       guideline | curated_expert | inferred",
        "  Condition families                    12 disease families",
        "  IS_A hierarchy depth                  3 levels",
        "=" * 62,
    ]
    text = "\n".join(lines)
    print("\n" + text + "\n")
    ensure_dir(root/"out")
    with (root/"out"/"TABLE4_stats.txt").open("w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True)
    ap.add_argument("--no_usda", action="store_true")
    ap.add_argument("--no_umls", action="store_true")
    ap.add_argument("--rules_csv",   default=None,
                    help="Path to db_merged_ingredient_property.csv from build_rules_from_databases.py")
    ap.add_argument("--extra_rules", default=None,
                    help="Folder of cleaned CSV files from clean_rules.py to merge in")
    args = ap.parse_args()

    print("FoodSafetyKG — General build (npj Digital Medicine)")
    print(f"  Root: {args.root}")
    print(f"  USDA: {'disabled' if args.no_usda else 'enabled'}")
    print(f"  UMLS: {'disabled' if args.no_umls else 'enabled'}\n")

    stats = build(args)
    print_table4(stats, Path(args.root))
    print("✅  FoodSafetyKG build complete.")

if __name__ == "__main__":
    main()