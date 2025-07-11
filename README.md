# Log Linear Attention

![Figure](figs/recurrent.png)

## Setup

1. Clone the repository and its submodules:
```bash
git clone --recursive https://github.com/HanGuo97/log-linear-attention.git
cd log-linear-attention
```
1.1.

```
conda create -n fla python=3.10
conda activate fla
conda install nvidia/label/cuda-12.6.1::cuda
```

2. Install the package and its dependencies:
```bash
pip install -e . # fails, but it's okay
pip install -e flame/
pip install -r flame/3rdparty/torchtitan/requirements.txt
pip install -e . # repetition is intended because the first pip install fails, this one also fails though))
pip install git+https://github.com/Dao-AILab/causal-conv1d.git
pip install git+https://github.com/state-spaces/mamba.git
pip install transformers==4.45.0 jaxtyping zstandard
pip install flash-attn --no-build-isolation
```

### Docker Installation (Not needed)

We provide a `Dockerfile` for containerized setup. To use it:

```bash
# Build the Docker image
DOCKER_BUILDKIT=1 docker build \
    -t log-linear-attention \
    -f Dockerfile \
    .

# Run the container
docker run -ti \
    --gpus all \
    log-linear-attention \
    bash
```

## Data Preparation

Run the preprocessing script:
```bash
DATASET_PATH=<your_path_to_save_dataset>
python -m hattention.preprocess_data $DATASET_PATH
python -m hattention.convert_to_parquet $DATASET_PATH $DATASET_PATH-parquet
```

> [!NOTE]
> The data preprocessing step may take a while (~2-3 hours).

## Training

1. Navigate to the training framework:
```bash
cd flame/
```

2. Launch training with the following command:
```bash
bash ../scripts/train_flame_longdata.sh --name [NAME] --config [CONFIG] --seed [--ac]
```

- `NAME`: Name for the experiment and save path
- `CONFIG`: Name of the config file in `configs/flame/` (without .json extension)
- `--seed`: Create a seed checkpoint before training
- `--ac`: Optional flag to enable activation checkpointing

For example:

```bash
bash ../scripts/train_flame_longdata.sh --name test --config transformer_mid2 --seed 42
```
> [!NOTE]
> 1. Before training, modify the absolute file paths in `scripts/train_flame_longdata.sh` to match your setup (add your $DATASET_PATH-parquet path in the script, adjust the number of gpus, batch_size)
> 2. The first training step will compile Triton kernels, which may take tens of minutes
> 3. batch_size = 2 takes around 35GiB VRAM on 2 gpus. So on 80 GiB GPU you can probably use either 4 or 8 for faster training.

# Acknowledgement
Special thanks to Tianyuan Zhang, Jyo Pari, Adam Zweiger, and Yu Zhang for lots of help and discussions.