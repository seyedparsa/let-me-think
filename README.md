# bepatient

## Running the code
1. Generate the dataset
```bash
sh scripts/gen.sh
```
2. Train the model
```bash
sh scripts/train.sh
```
3. Evaluate the model
```bash
sh scripts/eval.sh
```

## Repository Structure Overview
### `configs/`
- `training_config.yaml`: Training configuration.
- `model_config.yaml`: Model configuration.
- `accelerate.yaml`: Accelerator configuration.

### `scripts/`
- `gen.sh`: Generates a dataset of given graph, search, and task types.
- `train.sh`: Trains a model based on the given training configuration.
- `eval.sh`: Evaluates the performance of the models.

### `.`
- `gen.py`: Dataset generation logic.
- `train.py`: Training logic.
- `eval.py`: Evaluation logic.
- `mission.py`: Mission class definition.
- `utils.py`: Helpful methods.
