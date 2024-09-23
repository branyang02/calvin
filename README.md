# Calvin

This is a wrapper repo of the original [calvin repo](https://github.com/mees/calvin) where everything just works.


## Installation

1. Create conda environment
```bash
conda env create -f environment.yml && conda activate calvin-env
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

> **_NOTE:_**  Installation may take a few minutes due to local package dependencies in `calvin-sim`.

## Usage

Navigate to `eval-calvin` and run the following command to evaluate the performance of Calvin on the given dataset.

```bash
python eval.py --dataset_path <path_to_dataset>
```

For all available options, run with the `-h` flag.
