# Regression-based-Analytic-Incremental-Learning
Official implementation of [Advancing Cross-domain Discriminability in Continual Learning of Vision-Language Models](https://arxiv.org/pdf/2406.18868)

The paper has accepted by NeurIPS 2024.

## Installation

Create a conda environment and install dependencies:

```bash
git clone https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning.git
cd RAIL

conda create -n rail python=3.8
conda activate tip_adapter

pip install -r requirements.txt
```

## Data Preparation

We suggest putting all required datasets under the folder
```bash
RAIL/
|-- datasets/
```

Please refer to the following guides for setting up datasets:
[CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)

## Running

You may set the dataset sequence and other hyper-parameters in the config file [analytic_clip.yaml](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning/blob/master/configs/analytic_clip.yaml)

We have developed two forms of the RAIL method.
For the primal RAIL:
```bash
python primal_RAIL.py
```
For the dual RAIL:
```bash
python dual_RAIL.py
```

## Acknowledgement

Our repo benefits from [CLIP](https://github.com/openai/CLIP) and [CoOp](https://github.com/KaiyangZhou/CoOp). We thank them for their wonderful works.

---
