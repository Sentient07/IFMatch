# IFMatch

PyTorch implementation for paper [Implicit field supervision for robust non-rigid shape matching](https://arxiv.org/abs/2203.07694)

## Setup

1. Setup the conda environment using from the `env.yml` as `conda env create -f env.yml` and activate it.

2. Install the `pytorch-meta` as `cd pytorch-meta && python setup.py install`

## Dataset and Pre-Processing

1. We provide the datasets and variants used in our paper here: https://nuage.lix.polytechnique.fr/index.php/s/RjtprA8gSjgnEkt 

2. Once the dataset have been downloaded, we have two-staged pre-processing,

    1. Sampling SDF: To sample points with SDF, we follow the DeepSDF scheme as given [here](https://github.com/facebookresearch/DeepSDF#pre-processing-the-data). Assume all the pre-processed `npz` files are placed into `/path/to/npz`.

    2. Sampling surface with normal: For this, we use the [`mesh-to-sdf`](https://pypi.org/project/mesh-to-sdf/) package. To perform this step, run `data_process.py` the path to `ply` files and the `npz` files from previous point.

3. Once the pre-processing is done, your data directory should have three directories, `free_space_pts` containing the SDF, `surface_pts_n_normal` containing the surface points along with normal information and `vertex_pts_n_normal` containing vertex points.

4. Step 2 is repeated for both training and test dataset alike. 

## Training

To train, run the following by appropriately replacing parameters,

```
python train.py --config configs/train/<dataset>.yml --split split/train/dataset.txt --exp_name <my_exp>
```

## Evaluation

Our evaluation is two staged, first we find the optimal latent vector, then we solve for the P2P map between shapes

1. To solve for MAP,

```
python evaluate.py --config configs/eval/<dataset.yml>
```

2. To obtain the P2P map,

```
python run_matching.py --config configs/eval/<dataset.yml> --latent_opt
```

## Pre-trained Models

We perform 3 distinct training in total for reported results in the paper. Respective models can be downloaded from [here](https://nuage.lix.polytechnique.fr/index.php/s/PaLnSjAk9cjZtS4)

### Acknowledgements
We thank authors of DIF-Net and SIREN for graciously open-sourcing their code.
