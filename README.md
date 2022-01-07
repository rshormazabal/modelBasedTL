# Not from scratch: predicting thermophysical properties through model-based transfer learning using graph convolutional networks  

This repository is the official implementation of _"Predicting thermophysical properties through model-based transfer learning using graph convolutional networks"_.

<p align="center">
  <img src="/images/network1.png" />
</p>

An example of the architecture used in this work:

<p align="center">
  <img src="/images/model_example1.png" />
</p>

<p align="center">
  <img src="/images/model_example2.png" />
</p>

## Requirements

The implementation is written around the "Pytorch Geometric" library. Refer to [Pytorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the instalation process. Code tested on Ubuntu 18.04.
To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model in the paper, set parameters in the YAML file _parameters.YAML_.

```YAML
task_args
  type: 'experimental'
  name': 'critical_temperature'
preprocessing_args
  smiles_path:  './data/CAS_SMILES.csv'
  inference_file: None
  use_chirality:  True
  implicit_h: True
  n_jobs: 4
  root_path: './'
split_args
  split_mode: 'random'
  bench_threshold: None
  fold_n: None,
  test_size: 0.1
  random_state': 999
model_args
  node_features_dim: 26
  edge_features_dim': 11
  global_pooling_method: 'global_add'
  s2s_processing_steps: 3
  hidden_dim: 256
  aggr: 'mean'
  message_passing_steps': 3
  dropout: 0.3
  device': 'cuda'
optimizer_args
  lr: 5e-4
  weight_decay: 1e-3
training_args
  batch_size: 128
  epochs: 250
  val_size: 0.1
  n_bins: None
  random_state: 999
```

then run the command:

```python
python main.py  --train --input-data <path_to_data> --parameters <path_to_parameters_YAML>
```

## Evaluation

To evaluate the model on new data:

```python
python main.py --eval --input-data <path_to_data> --model-file <path_to_model>
```

To run benchmarks agaisnt other machine learning methods:

```python
python benchmarks.py --input-data <path_to_data> --model-file <path_to_model>
```

## Pre-trained Models

You can use pretrained models from **model-generated data**, aswell as models **fine-tuned** with experimental measurements. Both available in the models folder.

## Results

Our model achieves the following performance on :

### [Critical properties](<https://en.wikipedia.org/wiki/Critical_point_(thermodynamics)>)

|              |              |                 | **Mean absolute error** |               |                                            | **Mean absolute percentage error** |               |                                           |
| ------------ | ------------ | --------------- | :---------------------: | :-----------: | :----------------------------------------: | :--------------------------------: | :-----------: | :---------------------------------------: |
|              | **Property** | **Data points** |         **GCM**         | **GCNN-NOTL** |                **GCNN-TL**                 |              **GCM**               | **GCNN-NOTL** |                **GCNN-TL**                |
| **Training** |              | 642             |          8.43           |     4.52      |                    2.51                    |                1.35                |     0.73      |                   0.44                    |
|              |              | 333             |          69.35          |     42.01     |                   21.82                    |                2.11                |     1.29      |                   0.71                    |
|              |              | 247             |          6.78           |     2.36      |                    1.71                    |                1.98                |     0.74      |                   0.67                    |
| **Test**     |              | 211             |          78.01          |   **38.86**   | **<span style="color:blue">27.58</span>**  |               16.02                |   **8.37**    | **<span style="color:blue">5.76</span>**  |
|              |              | 219             |         429.61          |   **324.4**   | **<span style="color:blue">297.56</span>** |               14.16                |   **11.1**    | **<span style="color:blue">10.09</span>** |
|              |              | 96              |          65.72          |   **55.8**    | **<span style="color:blue">47.12</span>**  |                9.43                |   **6.26**    | **<span style="color:blue">5.41</span>**  |

Comparison with group-contribution methods for **experimental critical temperature**.

<p align="center">
  <img src="/images/results1.png" />
</p>
