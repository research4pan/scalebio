import json
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Convert HF data to LMFlow data')
parser.add_argument('--ds_name', type=str)
parser.add_argument('--save', type=str)

args = parser.parse_args()

ds = load_dataset(args.ds_name, split='train')
if 'conversations' in ds.column_names:
    ds = ds.rename_column('conversations', 'messages')

if not 'conversation_id' in ds.column_names:
    ds = ds.map(lambda x, idx: {'conversation_id': idx}, with_indices=True)

instances = []
for d in ds:
    instance={
        'conversation_id': d['conversation_id'],
        'messages': []
    }
    if d['messages'][0]['role'] == 'system':
        instance['system'] = d['messages'][0]['content']
        instance['messages'] = d['messages'][1:]
    else:
        instance['messages'] = d['messages']
    instances.append(instance)

lmflow_ds = {
    "type": "conversation",
    "instances": instances
}
json.dump(lmflow_ds, open(args.save, 'w'), indent=2, ensure_ascii=False)