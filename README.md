# Bag of Visual Words

This is the [2020 Spring: BoVW Competition](https://www.kaggle.com/c/2020backofwordrcv/overview) in RCV Sejong.

## Requriements

### Environment

- **OS**: Linux / macOS
- **Language**: Python >= 3.7
- **Packages**: see the [requirements.txt](./requirements.txt)
- **Kaggle API token**: see a [guide](https://github.com/Kaggle/kaggle-api#api-credentials) about it

### Steps for preparing

1. Install Python packages

```sh
$ pip install -r requirements.txt
```

2. Download the dataset

```sh
$ make download
```

## Codes

```sh
$ python main.py [-f] [--image-size IMAGE_SIZE [IMAGE_SIZE ...]]
        [--verbose VERBOSE] [--ratio RATIO] [--seed SEED]
        [--voc-size VOC_SIZE] [-C REGULARIZATION]
        [--result-log RESULT_LOG] [--train] [--show] [--top-n TOP_N]

optional arguments:
  -h, --help            show this help message and exit
  -f, --force-train     Force training even if a codebook have
  --image-size IMAGE_SIZE [IMAGE_SIZE ...]  Resize image size | (0,0) means that do not resize
  --verbose VERBOSE     Verbose level of scikit-learn
  --ratio RATIO         Split ratio of train/validation
  --seed SEED           Random seed number
  --voc-size VOC_SIZE   Size of vocabulary
  -C REGULARIZATION, --regularization REGULARIZATION  Regularization parameter in LinearSVC
  --result-log RESULT_LOG   Log file name
  --train               Train and evaluate BoVW model
  --show                Show accuracy in the log file
  --top-n TOP_N         Show top N accuracy
```

### Train

```sh
$ python main.py --train
```

### Show Top-N accuracy

```sh
$ python main.py --show --top-n 10
```

## Issues

- Please check [here](./ISSUES.md)
