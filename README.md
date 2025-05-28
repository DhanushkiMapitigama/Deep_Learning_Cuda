# Deep_Learning_Cuda

This project provides an singularity container setup for developing and experimenting with **custom CUDA kernels for PyTorch neural networks**.

## Container Setup

To build the container in an Apptainer environment, run:

```bash
apptainer build DL_cuda.sif DL_cuda.def

## Development 

For interactive development with CUDA support:

```bash
apptainer exec --nv DL_cuda.sif bash

Inside the container, verify that CUDA is properly configured:

```bash
nvcc --version

## Experiments

For experiments (on HPC cluster):
    Slurm job instructions to be added.