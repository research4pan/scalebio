import json
import random

random.seed(42)

INPUT='/home/panxingyuan/bilevel_llm/data/alpaca_data.json'
OUT_TR='/home/panxingyuan/bilevel_llm/data/train'
OUT_VAL='/home/panxingyuan/bilevel_llm/data/val'
OUT_TE='/home/panxingyuan/bilevel_llm/data/test'
NUM_PARTITION=10

def shuffle_and_partition(data, k):
    # Step 1: Shuffle the data
    random.shuffle(data)
    
    # Step 2: Select top 10% of the data
    top_10_percent_cutoff = max(1, len(data) // 10)
    top_10_percent = data[:top_10_percent_cutoff]
    
    json.dump(top_10_percent, open(OUT_VAL+'/val.json', 'w'), indent=4, ensure_ascii=False)

    top_20_percent_cutoff = max(1, len(data) // 5)
    top_20_percent = data[top_10_percent_cutoff:top_20_percent_cutoff]
    
    json.dump(top_20_percent, open(OUT_TE+'/test.json', 'w'), indent=4, ensure_ascii=False)

    # Remaining data after removing top 10%
    remaining_data = data[top_20_percent_cutoff:]
    
    # Step 3: Split the remaining data into K partitions
    partitions = []
    partition_size = max(1, len(remaining_data) // k)
    for i in range(0, len(remaining_data), partition_size):
        partitions.append(remaining_data[i:i + partition_size])
    
    # Adjust the last partition to include any leftover items
    if len(partitions) > k:
        last_partition = partitions[-2] + partitions[-1]
        partitions = partitions[:-2] + [last_partition]
    
    # Saving partitions (for demonstration, we'll just print them)
    for i, partition in enumerate(partitions):
        json.dump(partition, open(OUT_TR+f'/tr_{i}.json', 'w'), indent=4, ensure_ascii=False)

all=json.load(open(INPUT))
shuffle_and_partition(all, k=10)

