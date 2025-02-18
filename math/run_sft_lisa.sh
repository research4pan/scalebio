

CUDA_VISIBLE=0,1,2,3,4,5,6,7

hf_ds=HanningZhang/scalebio_qwen32_math_20k_baseline
hf_val_ds=pxyyy/rlhflow_scalbio_test
model_and_tok=meta-llama/Meta-Llama-3-8B
# model_and_tok=google/gemma-2-9b
conv_template=llama3

hf_ds_str=$(echo ${hf_ds}|sed 's/\//-/g')
tmp_data_dir=./tmp_data/${hf_ds_str}/
val_data_dir=./tmp_data/${hf_ds_str}_val/
mkdir -p ${tmp_data_dir}
mkdir -p ${val_data_dir}
python3 hf2lmflow.py --ds_name ${hf_ds} --save ${tmp_data_dir}/data.json
python3 hf2lmflow.py --ds_name ${hf_val_ds} --save ${val_data_dir}/data.json

model_str=$(echo ${model_and_tok}|sed 's/\//-/g')

lisa_activated_layers=2
lisa_interval_steps=20
gradient_accumulation_steps=2
per_device_train_batch_size=8
epoch=1
project_dir=test
for lr in 2e-5
do
    # Finetune
    exp_id=scalebio-scalebio-${model_str}-${hf_ds_str}-${epoch}-$lr-nolisa-bs128-gemma-uniform #-lisa_${lisa_activated_layers}_${lisa_interval_steps}
    # project_dir=$(cd "$(dirname $0)"; pwd)
    log_dir=${project_dir}/log/${exp_id}
    output_dir=${project_dir}/output_models/${exp_id}
    
    echo $exp_id
    
    mkdir -p ${output_dir} ${log_dir}

    export TRANSFORMERS_VERBOSITY=info

    deepspeed --master_port=7964 --include=localhost:${CUDA_VISIBLE} finetune.py \
        --model_name_or_path ${model_and_tok} \
        --trust_remote_code 1 \
        --dataset_path ${tmp_data_dir}/ \
        --eval_dataset_path ${val_data_dir}/ \
        --output_dir ${output_dir} --overwrite_output_dir \
        --conversation_template ${conv_template} \
        --num_train_epochs $epoch \
        --learning_rate $lr \
        --use_lisa 1 \
        --disable_group_texts 1 \
        --block_size 1024 \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size 1 \
        --bf16 \
        --deepspeed configs/ds_config_zero3.json \
        --torch_dtype bfloat16 \
        --run_name ${exp_id} \
        --optim adamw_torch_fused \
        --logging_steps 1 \
        --do_train \
        --do_eval \
        --ddp_timeout 72000 \
        --save_total_limit 1 \
        --load_best_model_at_end False \
        --eval_steps 20 \
        --save_only_model \
        --evaluation_strategy "steps" \
        --dataloader_num_workers 1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --gradient_checkpointing True \
        --use_flash_attention 1 \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --lisa_activated_layers ${lisa_activated_layers} \
        --lisa_interval_steps ${lisa_interval_steps} \
        | tee ${log_dir}/train.log \
        2> ${log_dir}/train.err
done