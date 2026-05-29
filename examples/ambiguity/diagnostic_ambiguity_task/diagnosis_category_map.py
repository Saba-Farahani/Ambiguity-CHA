from pathlib import Path

out_path = Path("data/diagnosis_category_map.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

csv_text = """diagnosis,category
Acute bacterial sinusitis,sinusitis
Acute bronchitis,bronchitis
Acute infective cystitis,uti
Acute viral pharyngitis,pharyngitis
Alzheimer's disease,neurocognitive
Anemia,hematologic
Atopic dermatitis,dermatologic
Bleeding from anus,gastrointestinal
Child attention deficit disorder,neurodevelopmental
Childhood asthma,asthma
Chronic congestive heart failure,cardiovascular
Chronic intractable migraine without aura,migraine
Chronic obstructive bronchitis,bronchitis_copd
Chronic obstructive bronchitis (disorder),bronchitis_copd
Contact dermatitis,dermatologic
Familial Alzheimer's disease of early onset,neurocognitive
Fibromyalgia,pain_rheumatologic
Idiopathic atrophic hypothyroidism,endocrine
Impacted molars,dental_or_oral
Inflammatory disorder due to increased blood urate level,gout
Ischemic heart disease,cardiovascular
Opioid abuse,substance_use
Osteoarthritis of hip,musculoskeletal
Osteoarthritis of knee,musculoskeletal
Perennial allergic rhinitis,rhinitis
Perennial allergic rhinitis with seasonal variation,rhinitis
Primary fibromyalgia syndrome,pain_rheumatologic
Pulmonary emphysema,bronchitis_copd
Pyelonephritis,uti
Recurrent urinary tract infection,uti
Renal dysplasia,renal
Rheumatoid arthritis,pain_rheumatologic
Seasonal allergic rhinitis,rhinitis
Sinusitis,sinusitis
Sleep disorder,sleep
Streptococcal sore throat,pharyngitis
Viral sinusitis,sinusitis
Acute pharyngitis,pharyngitis
Acute bacterial pharyngitis,pharyngitis
Tonsillitis,pharyngitis
Strep throat,pharyngitis
Streptococcal pharyngitis,pharyngitis
Viral pharyngitis,pharyngitis
Infectious mononucleosis,pharyngitis
Acute sinusitis,sinusitis
Bacterial sinusitis,sinusitis
Acute bronchitis (disorder),bronchitis
COPD,bronchitis_copd
Chronic obstructive pulmonary disease,bronchitis_copd
Allergic rhinitis,rhinitis
Allergic rhinitis caused by pollen,rhinitis
Common cold,rhinitis
Viral upper respiratory infection,rhinitis
Upper respiratory tract infection,rhinitis
Asthma,asthma
Influenza,viral_respiratory
COVID-19,viral_respiratory
Viral gastroenteritis,gastrointestinal
Gastroenteritis,gastrointestinal
Acute viral rhinosinusitis,sinusitis
Acute Viral Rhinosinusitis,sinusitis
"""

out_path.write_text(csv_text, encoding="utf-8")
print(f"Wrote: {out_path.resolve()}")