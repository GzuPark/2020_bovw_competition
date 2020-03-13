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

## Run

```sh
# Default argument values
$ make run

# Optional values
$ python main.py [-f] [--verbose VERBOSE] [--ratio RATIO] [--seed SEED] [--voc-size VOC_SIZE] [-C REGULARIZATION]

optional arguments:
  -f, --force-train     Force training even if a codebook have
  --verbose VERBOSE     Verbose level of scikit-learn
  --ratio RATIO         Split ratio of train / validation
  --seed SEED           Random seed number
  --voc-size VOC_SIZE   Size of vocabulary
  -C REGULARIZATION, --regularization REGULARIZATION    Regularization parameter in LinearSVC
```



## Issues

- Please check [here](./ISSUES.md)
