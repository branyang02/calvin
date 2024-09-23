# Calvin

This is a wrapper repo of the original [calvin repo](https://github.com/mees/calvin) where everything just works.


https://github.com/user-attachments/assets/7cb096f9-243c-4a6e-893d-a56eb3194260


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

To add your own model, simply modify the `CustomModel` class in `eval.py` as shown below. Functionalities are defined in `calvin_agent.models.calvin_base_model.CalvinBaseModel`.

```python
class CustomModel(CalvinBaseModel):
    def __init__(self):
        # TODO: Add any model specific initialization here
        pass

    def reset(self):
        # TODO: Add any model specific reset here
        pass

    def step(self, obs, goal):
        # TODO: Add your model's logic here

        # Random action
        action_displacement = np.random.uniform(low=-1, high=1, size=6)
        action_gripper = np.random.choice([-1, 1], size=1)
        action = np.concatenate((action_displacement, action_gripper), axis=-1)
        return action
```
