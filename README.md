
# MLTools

MLTools is a Python package providing a suite of utilities for machine learning, including tools for data processing, training with distributed computing resources, and utilities for working with *.zarr data cubes.

## Installation

You can install MLTools directly from PyPI:
```bash
pip install MLTools
```

Make sure you have Python version 3.8 or higher.

## Features

- Data preprocessing and normalization functions
- Distributed training framework compatible with PyTorch
- Utilities for working with ML data structures, such as datasets and data loaders

## Usage

To use MLTools in your project, simply import the necessary module:

```python
from MLTools.data_processing import normalize, standardize
from MLTools.torch_training import train_one_epoch
# Other imports...
```

You can then call the functions directly:

```python
# Normalizing data
normalized_data = normalize(your_data, data_min, data_max)

# Training a model for one epoch
model, train_pred, last_loss = train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer, device)
```

## License

MLTools is released under the MIT License. See the [LICENSE](https://github.com/Julia310/MLTools/blob/main/LICENSE) file for more details.