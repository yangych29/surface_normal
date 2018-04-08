# Single-view Surface Normal Prediction
This is the course project of EECS 442 Computer Vision (2018 Winter), University of Michigan.

## Dependencies

The code is tested on python3.6. Related packages include

- pytorch
- opencv
- numpy
- scipy
- imageio
- tqdm

## Training

To set up,

```bash
mkdir exp
```

To start training a new model,

```bash
python train.py -e sn_full -t sn
```

To continue training model `sn_full`,

```bash
python train.py -c sn_full -e sn_full -t sn
```

The training code would automatically save `${model}_${epoch}` under `exp`. For example, if we train a model `sn_full` for 10 epochs, there would be `sn_full_1`, `sn_full_2`, etc. under `exp`. These snapshots are used for validation.

## Evaluation

Evaluation is a separate pipeline. Firstly, we need generate the predictions.

```bash
rm -rf save
mkdir save
python generate.py -c sn_full -e sn_full -t sn
```

Then we use evaluation code to calculate mean angle error (MAE).

```bash
python evaluate.py -p save -g ~/datasets/eecs442challenge/train/normal/
```

These can be done directly by `test.sh`.

## Ensemble

Besides training and evaluation, we want to submit an ensemble of ConvNets to improve performance. These can be done by

```bash
python ensemble.py
```

## Reference

We use a lot of code from [umich-vl/pose-ae-train](https://github.com/umich-vl/pose-ae-train).