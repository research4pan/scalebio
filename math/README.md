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
