# Feature Drift Oriented Distribution Reconstruction for Imbalanced Class Incremental Learning (ECAI 2025)

This repo is the official implementation of ECAI 2025 paper: Feature Drift Oriented Distribution Reconstruction for Imbalanced Class Incremental Learning.
> Feature Drift Oriented Distribution Reconstruction for Imbalanced Class Incremental Learning
>
> Tingmin Li, Fengqiang Wan, Yipeng Lin, Yang Yang

**Key words: class incremental learning, imbalance learning .**


## Requirements

- Python 3.8.0

```setup
pip install -r requirements.txt
```

## Data Preparation
We follow prior works to conduct experiments on three standard datasets: CIFAR100, ImageNet100, and ImageNet1000.
### Download Datasets
- **CIFAR100** dataset will be downloaded automatically to the directory `./data`.

- **ImageNet100** and **ImageNet1000** datasets cannot be downloaded automatically, you should specify the folder of your dataset in `utils/data.py`.

```python
  def download_data(self):
        data_path = ""
        assert data_path, "please specify the data path "
        train_dir = '[data_path]/train'
        test_dir = '[data_path]/val'
```

## Training scripts

- Train CIFAR100 B0 10steps

  ```
  python main.py --config=./configs/cifar100/B0_10steps.json
  ```
- Train ImageNet100 B0 10steps

  ```
  python main.py --config=./configs/imagenet100/B0_10steps.json
  ```
- Train ImageNet1000 B0 10steps

  ```
  python main.py --config=./configs/imagenet1000/B0_10steps.json
  ```


The configuration files for other settings can be found in `./exps`.


## Acknowledgement
This repository is developed mainly based on the PyTorch implementation of [MAFDRC](https://github.com/chen-xw/DRC-CIL). Many thanks to its contributors!

## Citation
If you found our codebase useful, please consider citing our paper:

    @inproceedings{li2025fddr,
      title={Feature Drift Oriented Distribution Reconstruction for Imbalanced Class Incremental Learning},
      author={Tingmin Li, Fengqiang Wan, Yipeng Lin, Yang Yang},
      booktitle={The 28th European Conference on Artificial Intelligence},
      year={2025},
    }