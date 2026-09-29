import pandas as pd
import json
  
from pathlib import Path
from typing import List, Dict

class TropicalDiseaseLoader:
    def __init__(self, root_path: str):
        # create a Path object where the disease folders will be stored
        self.root_path = Path("Demos/tropical_diseases_dataset")
        
        self.diseases = ['dengue', 'malaria', 'leptospirosis','melioidosis', 
                        'neurocysticercosis', 'scrub_typhus', 'typhoid', 'zika']
    
    def load_cases(self) -> List[Dict]:
        all_cases = []
        
        for disease in self.diseases:
            # joins the root path to each disease folder
            disease_path = self.root_path / disease

            # skip missing disease folders
            if not disease_path.exists():
                continue
            # try cases.csv first then cases_without_image.csv
            cases_file = disease_path / 'cases.csv'
            if not cases_file.exists():
                cases_file = disease_path / 'cases_without_image.csv'
            if not cases_file.exists():
                continue
            
            # load cases description of a disease
            cases_df = pd.read_csv(disease_path / 'cases.csv')
            
            # load images
            image_meta = {}
            img_file = disease_path / 'image_metadata.json'
            if img_file.exists():
                with open(img_file) as f:
                    for line in f:
                        item = json.loads(line)
                        if 'case_id' in item:
                            if item['case_id'] not in image_meta:
                                image_meta[item['case_id']] = []
                            image_meta[item['case_id']].append(item)
            
            # combine cases and images
            for _, row in cases_df.iterrows():
                case_dict = row.to_dict()
                case_dict['disease'] = disease
                case_dict['image_meta'] = image_meta.get(case_dict['case_id'], [])
                all_cases.append(case_dict)
        
        return all_cases

# loader execution
loader = TriopicalDiseaseLoader('Demos/tropical_diseases_dataset')
all_cases = loader.load_cases()
print(f"Loaded {len(all_cases)} cases from {len(loader.diseases)} diseases")
