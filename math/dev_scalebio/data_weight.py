import json
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
NUM=20_000
cnt=0
def filter_data(example):
    messages = example['messages']
    global cnt
    if len(messages) < 2:
        cnt+=1
        return False
    for i in range(len(messages)):
        if messages[0]['role'] == 'system':
            if i == 0:
                turn = 'system'
            elif i % 2 == 1:
                turn = 'user'
            else:
                turn = 'assistant'
        else:
            if i % 2 == 0:
                turn = 'user'
            else:
                turn = 'assistant'

        if messages[i]['role'] != turn:
            cnt+=1
            return False
        if len(messages[i]['content']) == 0:
            cnt+=1
            return False
    return True

def convert_to_sharegpt_format(dataset):
    res=[]
    for d in dataset:
        items=[]
        for c in d['messages']:
            if c['role']=='user':
                role='human'
            elif c['role']=='assistant':
                role='gpt'
            else:
                role='system'
            items.append({
                'value': c['content'],
                'from': role
            })
        res.append({
            'items': items
        })
    return res



weight = {
'SlimOrca':	0.11364,
'dart-math-uniform':	0.06575,
'GPT4-LLM-Cleaned':	0.14448,
'MathInstruct':	0.19033,
'GPTeacher-General-Instruct':	0.13023,
'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry':	0.07001,
'UltraInteract_sft':	0.08076,
'WizardLM_evol_instruct_V2_196k':	0.07469,
'Magicoder-Evol-Instruct-110K':	0.06935,
'orca-math-word-problems-200k':	0.06077,
}

print(sum(weight.values()))

ds_list = []

for k, v in weight.items():
    smp_num = int(NUM * v)
    ds = load_dataset(f"pxyyy/{k}")
    ds = ds['train'].filter(filter_data)
    print(f'for ds: {k}, sample num: {smp_num}, total num: {len(ds)}')
    assert smp_num <= len(ds)
    ds = ds.shuffle(seed=42).select(range(smp_num))
    ds_list.append(ds)

ds = datasets.concatenate_datasets(ds_list)
print(f'final total num: {len(ds)}')
if len(ds) < NUM:
    print('Not enough data')
    smp = NUM - len(ds)
    extra = ds.shuffle(seed=42).select(range(smp))
    ds = datasets.concatenate_datasets([ds, extra])

with open("exp_rlhflow_less_oss/val.json",'r') as f:
    val = json.load(f)
    
new_ds =[]
for i,sample in enumerate(val):
    tmp = {}
    mess = [{"role":"user","content":sample['items'][0]['value']},
            {"role":"assistant","content":sample['items'][1]['value']}]
    tmp['messages'] = mess
    tmp['conversation_id'] = 99999+ i 
    new_ds.append(tmp)
    
new_ds = Dataset.from_list(new_ds)
ds = concatenate_datasets([ds,new_ds])
ds = ds.shuffle(seed=42)
print(len(ds))

ds.push_to_hub("HanningZhang/scalebio_llama_math_20k_uw2e-6_alpha100_lambda1e-2")
