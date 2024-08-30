# M3DA-exp

## Install

```
git clone https://github.com/NeurIPS2024DB1854/M3DA-exp.git
cd M3DA-exp && pip install -e .
```

## Getting started

To run an experiment, use the following commands:

```
dpipe-build path/to/M3DA-exp/configs/experiments/wmgmcsf/dann.config path/to/experiments/wmgmcsf/dann
cd path/to/experiments/wmgmcsf/experiment_0
exec lazycon run ../resources.config run_experiment
```

Experimental configs are organized by the dataset names inside `configs/experiments`.
Running `exec lazycon run` should be repeated for every sub-fold (experiment_i) of the created exp.
