from pathlib import Path
import json
from 3_data_structurer import CyberDataStructurer

structurer = CyberDataStructurer(ollama_port=11434)
path = Path('filtered_data/arxiv_papers_20251011_073608_filtered_20251011_143145.json')
data = json.loads(path.read_text())
entries = structurer.flatten_entries(data)
print('flattened entries:', len(entries))
for idx, entry in enumerate(entries[:3], 1):
    print('\nEntry', idx)
    if isinstance(entry, dict):
        print('keys:', list(entry.keys()))
        print('entry id:', entry.get('id'))
        print('has arxiv_primary_category:', 'arxiv_primary_category' in entry)
    else:
        print('non-dict entry:', type(entry), entry)
