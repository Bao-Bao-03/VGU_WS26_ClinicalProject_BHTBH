from RAG.data.loader import TropicalDiseaseLoader

def test_load_cases():
    loader = TropicalDiseaseLoader("Demos/tropical_diseases_dataset")
    cases = loader.load_cases()
    # Basic assertions: returns a list and has at least one case
    assert isinstance(cases, list)
    assert len(cases) > 0
    # check first case has expected keys
    keys = set(cases[0].keys())
    assert 'disease' in keys
    assert 'case_id' in keys or 'image_meta' in keys
