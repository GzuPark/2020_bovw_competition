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

## How to use

```sh
$ python main.py [-h] [-f] [--config] [--seed SEED] [--ratio RATIO]
                 [--image-size IMAGE_SIZE [IMAGE_SIZE ...]]
                 [--feat-step-size FEAT_STEP_SIZE]
                 [--dense-step-size DENSE_STEP_SIZE] [-L LEVEL]
                 [--verbose VERBOSE] [--voc-size VOC_SIZE] [-C REGULARIZATION]
                 [--result-log RESULT_LOG] [--train] [--show] [--top-n TOP_N]

optional arguments:
    -h, --help            show this help message and exit
    -f, --force-train     Force training even if a codebook have
    --config              Configuration hyperparameter candidates
    --seed SEED           Random seed number
    --ratio RATIO         Split ratio of train/validation
    --image-size IMAGE_SIZE [IMAGE_SIZE ...]
                          Resize image size | (0,0) means that do not resize
    --feat-step-size FEAT_STEP_SIZE
                          Step size when extracting features
    --dense-step-size DENSE_STEP_SIZE
                          Step size when Dense SIFT
    -L LEVEL, --level LEVEL
                          Levels of SPM
    --verbose VERBOSE     Verbose level of scikit-learn
    --voc-size VOC_SIZE   Size of vocabulary
    -C REGULARIZATION, --regularization REGULARIZATION
                          Regularization parameter in LinearSVC
    --result-log RESULT_LOG
                          Log file name
    --train               Train and evaluate BoVW model
    --show                Show accuracy in the log file
    --top-n TOP_N         Show top N accuracy
```

### Train

```sh
$ python main.py --train
```

### Configuration

1. Create `config.py` like below example:
    ```python
    # config.py
    import numpy as np

    class Params():
        img_size = [[0, 0]]
        voc_size = [200]
        feat_step_size = [16]
        dense_step_size = [8]
        level = [2, 3]
        regularization = np.arange(0.01, 0.11, 0.01)
    ```

2. Train with configuration
    ```sh
    $ python main.py --train --config
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
