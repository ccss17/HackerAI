from collections import Counter
from pathlib import Path
import json

base = Path(__file__).resolve().parent / 'filtered_data'
summaries = {}

def collect_keys_from_list(entries, max_entries=10):
    key_counter = Counter()
    container_info = {}

    for entry in entries[:max_entries]:
        if isinstance(entry, dict):
            key_counter.update(entry.keys())
            for key, value in entry.items():
                if isinstance(value, dict):
                    container_info.setdefault(key, {'type': 'dict', 'keys': set()})
                    container_info[key]['keys'].update(value.keys())
                elif isinstance(value, list):
                    container_info.setdefault(key, {'type': 'list', 'elem_types': Counter(), 'elem_keys': set()})
                    container_info[key]['elem_types'][type(value[0]).__name__ if value else 'unknown'] += 1
                    if value and isinstance(value[0], dict):
                        container_info[key]['elem_keys'].update(value[0].keys())
        else:
            container_info.setdefault('non_dict_entries', set()).add(type(entry).__name__)

    top_keys = list(key_counter.keys())
    containers = {}
    for key, info in container_info.items():
        if isinstance(info, dict) and info.get('type') == 'dict':
            containers[key] = {
                'container_type': 'dict',
                'keys': sorted(info['keys'])
            }
        elif isinstance(info, dict) and info.get('type') == 'list':
            containers[key] = {
                'container_type': 'list',
                'element_types': dict(info['elem_types']),
                'element_keys': sorted(info['elem_keys']) if info['elem_keys'] else []
            }
        else:
            containers[key] = sorted(info)

    return top_keys, containers

for path in sorted(base.glob('*.json')):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        summaries[path.name] = {'error': str(exc)}
        continue

    file_summary = {}

    if isinstance(data, list):
        file_summary['root_type'] = 'list'
        file_summary['entry_count'] = len(data)
        if data:
            top_keys, containers = collect_keys_from_list(data)
            file_summary['root_keys'] = top_keys
            if containers:
                file_summary['containers'] = containers
    elif isinstance(data, dict):
        file_summary['root_type'] = 'dict'
        file_summary['root_keys'] = list(data.keys())
    else:
        file_summary['root_type'] = type(data).__name__

    summaries[path.name] = file_summary

import json
print(json.dumps(summaries, indent=2, sort_keys=True))
