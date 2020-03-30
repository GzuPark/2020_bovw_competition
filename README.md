# Bag of Visual Words

This is the [2020 Spring: BoVW Competition](https://www.kaggle.com/c/2020backofwordrcv/overview) in RCV Sejong.

## Requriements

### Environment

- **OS**: Ubuntu 18.04
- **Language**: Python >= 3.7
- **GPU**: CUDA >= 10.1, CUDNN >= 7.6
- **Packages**: see the [requirements.txt](./requirements.txt)
- **Kaggle API token**: see a [guide](https://github.com/Kaggle/kaggle-api#api-credentials) about it, save to assets directory
- **Docker**: NVIDIA-Docker

### Steps for preparing

1. Download the dataset

```sh
$ make download
```

2. Build docker images (NVIDIA-Docker required)

```sh
$ make docker-base
$ make docker-user PASSWORD=${PASSWORD_FOR_NOTEBOOK}
```

3. Run docker container

```sh
$ make docker-run CONTAINER_NAME=${CONTAINER_NAME} \
                  TENSORBOARD=${TENSORBOARD_PORT} \
                  NOTEBOOK=${NOTEBOOK_PORT} \
                  PORT=${SSH_PORT} \
                  POS_GPUS=${POS_GPUS}
```

4. Add `kaggle.json` file to `~/.kaggle/` for executing kaggle application

## How to use

```sh
$ python main.py [-h] [-f] [--seed SEED] [--ratio RATIO]
                 [--image-size IMAGE_SIZE [IMAGE_SIZE ...]] [--batch BATCH]
                 [--epochs EPOCHS] [--lr LR] [--optimizer OPTIMIZER]
                 [--network NETWORK] [--freeze] [--spp]
                 [--result-log RESULT_LOG] [--train] [--show] [--top-n TOP_N]

optional arguments:
  -h, --help            show this help message and exit
  -f, --force           Force run everything
  --seed SEED           Random seed number
  --ratio RATIO         Split ratio of train/validation
  --image-size IMAGE_SIZE [IMAGE_SIZE ...]
                        Resize image size
  --batch BATCH         Batch size
  --epochs EPOCHS       Epochs
  --lr LR               Learning rate
  --optimizer OPTIMIZER
                        Optimizer
  --network NETWORK     Network
  --freeze              Freeze pretrained parameters
  --spp                 Apply Spatial Pyramid Pooling
  --result-log RESULT_LOG
                        Log file name
  --train               Train and evaluate BoVW model
  --show                Show accuracy in the log file
  --top-n TOP_N         Show top N accuracy
```

### Train

```sh
$ python main.py --train

# Train with specific GPU devices
$ CUDA_VISIBLE_DEVICES=${GPUS} python main.py --train
```

### Show Top-N accuracy

```sh
$ python main.py --show --top-n 10
```

## Competitions

### Submit

```sh
$ make submit SUBMIT=${CSV_FILE_NAME} MSG=${SUBMIT_COMMENT}
```

### Check the leaderboard

```sh
$ make rank
```

## Issues

- Please check [here](./ISSUES.md)

## Authors

- [Jongmin Park](mailto:jmpark@rcv.sejong.ac.kr)
- [Hyunho Nam](mailto:namhh@rcv.sejong.ac.kr)

## Licence

[GPL-3.0](./LICENCE)
