# Scalebio for MATH

Here is the codebase for running Scalebio on Math tasks, followed by Supervised Fine-tuning and Evaluation.

## Running Scalebio for data reweighting

### Requirements for Scalebio

```
conda create -n scalebio python=3.10.9
conda activate scalebio

pip install torch==2.1.2 torchvision torchaudio
pip install transformers==4.43.4

pip install flash-attn==2.6.3
pip install accelerate==0.32.0
pip install peft==0.11.1
pip install datasets
pip install jsonlines

pip install wandb
```

### Running the code

```
cd dev_scalebio
bash run_scalebio.sh
```
You can run the Scalebio with LISA: 
```
bash run_scalebio_lisa.sh
```

- Please configure the `run_scalebio.sh` or `run_scalebio_lisa.sh` to set the model name and all the hyper-parameters.
- Please also configure the `fsdp_transformer_layer_cls_to_wrap` in `fsdp_config.yaml` for your own model.
- If you encounter **OSError: [Errno 39] Directory not empty:**, just ignore it and the code won't stop.

### Data Reweighting

- Please go to `wandb` and check the final weights in `samp prob.exp_rlhflow_less_oss`
- Copy the weights to the `weight` dictionary in `data_weight.py`, and also specify the **Huggingface path** to save the data.
```
python data_weight.py
```

## Running Supervised-Finetuning 

### Requirements

```
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```
### Running the code

```
bash run_sft.sh
```
You can also run the Supervised Fine-tuning with LISA:
```
bash run_sft_lisa.sh
```
