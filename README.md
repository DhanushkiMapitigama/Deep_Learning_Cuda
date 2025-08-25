# Deep_Learning_Cuda

This project provides an singularity container setup for developing and experimenting with **custom CUDA kernels for PyTorch neural networks**.

## Container Setup

To build the container in an Apptainer environment, run:

```bash
apptainer build DL_cuda.sif DL_cuda.def
```

## Development 

For interactive development with CUDA support:

```bash
apptainer exec --nv DL_cuda.sif bash
```

Inside the container, verify that CUDA is properly configured:

```bash
nvcc --version
```
### build custom cuda kernels with pip

To build and install custom CUDA kernels:
```bash
cd cuda_kernels
pip install --user -e .
```

## Build cuda kernals

You need to compile cuda kernals before using them

```bash
cd cuda_kernals
pip install --user -e .
cd ..
```

## To Run the model with custom functions

Configure the yaml file with preferred settings.

```bash
python main.py
```

## Experiments

To run performance experiments first give preffered model configurations in config.yaml.

Then run

```bash
python experiements/experiments.py
```

To do compute profiling experiments first give preffered data configurations in config.yaml.

Then run

```bash
python experiements/compute_experiments.py
```

## Tests

Checkout to test branch.

Run

```bash
pytest
```

or (to avoid specific wardnings if packages are outdated.)

```bash
pytest -W ignore::DeprecationWarning
```

