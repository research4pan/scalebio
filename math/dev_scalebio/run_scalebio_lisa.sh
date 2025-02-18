function main() {
    seed=0
    lr_u=2e-6
    lr_w=2e-6
    lr_lambda=1e-2
    batch_size=128
    micro_batch_size=1
    val_batch_size=1
    epoch=1
    exp_name=llama3_8b_scalebio_lisa
    # train data name: train_<name>.json
    data_dir=exp_rlhflow_less_oss
    num_partitions=`ls -l ${data_dir}/train_*.json|wc -l`
    log_dir=log_${exp_name}/
    save_dir=models/${exp_name}/
    wandb_proj='bilevel-optimization'
    alpha=100
    mkdir -p ${log_dir}
    mkdir -p ${save_dir}
    model_and_tok=meta-llama/Meta-Llama-3-8B
    # model_and_tok=meta-llama/Meta-Llama-3-8B
    model_str=$(echo ${model_and_tok}|sed 's/\//-/g')

    local exp_id=${exp_name}_lr-u-${lr_u}_lr-w-${lr_w}_lr_lambda-${lr_lambda}_seed-${seed}_alpha-${alpha}_epoch-${epoch}_model-${model_str}_nolisa-clip_model

          echo "$(date): ${exp_id}"

          accelerate launch --config_file fsdp_config.yaml python/main.py \
              --wandb_run_name ${exp_id} \
              --wandb_project ${wandb_proj} \
              --train-data ${data_dir}/train_\*.json \
              --val-data ${data_dir}/val.json \
              --test-data ${data_dir}/test.json \
              --model $model_and_tok \
              --lisa 1 \
              --micro_batch_size ${micro_batch_size} \
              --global_batch_size $batch_size \
              --max-length 1024 \
              --tokenizer-name $model_and_tok \
              --optimizer "name=adamw" \
              --init_lr 5e-4 \
              --clip_model \
              --minmax_init_lr_u ${lr_u} \
              --minmax_init_lr_w ${lr_w} \
              --minmax_init_lr_lambda ${lr_lambda} \
              --lr_scheduler "name=cosine_w_warmup, min_lr=0.0, warmup_ratio=0.03" \
              --validation_model_mode "train" \
              --minmax_validation_sampler stochastic \
              --sampler stochastic \
              --pseudo_random \
              --logging_conf_file conf/common.log_conf \
              --init_alpha ${alpha} \
              --tau 1 \
              --seed ${seed} \
              --epoch $epoch \
              --num_outer_iter 1 \
              --bf16 \
              --use_wandb \
              --val_batch_size ${val_batch_size} \
              --sharegpt_format \
              --eval_frequency 50 \
              --lisa_n_layers 4 \
              --response_loss_only \
              --num_partitions ${num_partitions} \
              --save_dir ${save_dir} \
            #   > ${log_dir}/${exp_id}.log \
            #   2> ${log_dir}/${exp_id}.err
}
# export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

main "$@"
            #   --response_loss_only \