import json
import random
import glob
import numpy as np

random.seed(42)

TR_DIR='/home/panxingyuan/bilevel_llm/data/train'
OUT_DIR='/home/panxingyuan/bilevel_llm/data/corrupted_train_no_out_2'
NUM_PARTITION=10

noise_rate=2.0*np.arange(NUM_PARTITION)/NUM_PARTITION

for idx in range(NUM_PARTITION):
    all=json.load(open(TR_DIR+f'/tr_{idx}.json'))
    print(TR_DIR+f'tr_{idx}.json', len(all))
    for a in all:
        if np.random.rand()<noise_rate[idx]:
            a['output']='.'
    json.dump(all, open(OUT_DIR+f'/tr_{idx}.json', 'w'), indent=4, ensure_ascii=False)


# for idx in range(NUM_PARTITION):
#     all=json.load(open(TR_DIR+f'/tr_{idx}.json'))
#     print(TR_DIR+f'tr_{idx}.json', len(all))
#     partition=int(noise_rate[idx]*len(all))
#     for k, a in enumerate(all[:partition]):
#         a['output']=all[(k+1)%partition]['output']
#     json.dump(all, open(OUT_DIR+f'/tr_{idx}.json', 'w'), indent=4, ensure_ascii=False)
