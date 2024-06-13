# Comparing Graph Transformers via Positional Encodings

Code for the ICML 2024 paper [Comparing Graph Transformers via Positional Encodings](https://arxiv.org/abs/2402.14202). 


## Instructions

### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb
pip install brec

conda clean --all
```

### Running an experiment with GraphGPS
```bash
conda activate graphgps

# Running an arbitrary config file in the `configs` folder
python main.py --cfg configs/<config_file>.yaml  wandb.use False
```
We provide the config files necessary to reproduce our experiments under `configs/` (see more below).

### W&B logging
To use W&B logging, set `wandb.use True` and have a `position_encoding` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).

To perform a hyperparameter search using W&B, see the instructions in the directory [wandb_hyperparameter_search](./wandb_hyperparameter_search/). 

## Citation
If you use this code, please cite our paper 

```bibtex
  @inproceedings{black2024comparing,
  title={Comparing Graph Transformers via Positional Encodings},
  author={Black, Mitchell and Wan, Zhengchao and Mishne, Gal and Nayyeri, Amir and Wang, Yusu},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
  }
```

as well as the previous papers that developed the original code for this repository

```bibtex
@article{muller2024attending,
title={Attending to Graph Transformers},
author={Luis M{\"u}ller and Mikhail Galkin and Christopher Morris and Ladislav Ramp{\'a}{\v{s}}ek},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024}
}
```
```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

