# Software.rizzuti2020EAGEtwri
# Time-domain implementation of the dual formulation of Wavefield Reconstruction Inversion

Julia implementation of the dual formulation of Wavefield Reconstruction Inversion

Additional technical information about this work can be found in the SLIM group page:
https://slim.gatech.edu/content/time-domain-wavefield-reconstruction-inversion-large-scale-seismics

This software release complements the submission to EAGE 2020

## Requirements

Currently supported Julia version: 1.3.1

### Package Dependencies: ###

This software is based on Devito (https://www.devitoproject.org/) and the Julia-Devito package JUDI (https://github.com/slimgroup/JUDI.jl). To install Devito and JUDI, follow the instructions included in the JUDI link.

To install other Julia packages, switch to package manager (using ']') in Julia's REPL) and type:

```
 add Optim
 add PyPlot
 add JLD2
```

## Instructions

To run the Gaussian lens inversion problem:

### Generate synthetic data

From the parent directory, type in Julia REPL:
include("./data/GaussLens/gendata_GaussLens.jl")

This will create the file "./data/GaussLens/GaussLens_data.jld" containing domain discretization details, true squared slowness model, source/receiver geometry, and synthetic data.

The squared slowness model can be inspected by typing:
imshow(model_true.m', aspect = "auto", cmap = "jet")

Shot gathers can be inspected by typing, e.g.,
imshow(dat.data[1], aspect = "auto", cmap = "jet")

### Run the TWRI dual inversion script

From the parent directory, type in Julia REPL:
include("./scripts/inversion_GaussLens.jl")

In order to run FWI on the same problem, comment/uncomment the relevant lines in the script.

## Authors

Software written by Gabrio Rizzuti (rizzuti.gabrio@gatech.edu) and Mathias Louboutin (mlouboutin3@gatech.edu), at Georgia Institute of Technology.
