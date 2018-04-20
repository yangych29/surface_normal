# Single-view Surface Normal Prediction
This is the course project of EECS 442 Computer Vision (2018 Winter), University of Michigan.

## Group Member

- Shengyi Qian ([@JasonQSY](https://github.com/JasonQSY))
- Linyi Jin ([@jinlinyi](https://github.com/jinlinyi))
- Yichen Yang ([@yangych29](https://github.com/yangych29))

## Demo

The left image is our network input, which is a gray-scale synthetic image. The right image is the network output, the color follows https://en.wikipedia.org/wiki/Normal_mapping

![input](demo/input.png)![pred](demo/pred.png)

## Dependencies

The code is tested on python3.6. Required packages include

- pytorch
- opencv
- numpy
- scipy
- imageio
- tqdm

It is only tested on Ubuntu 16.04 LTS with CUDA. But it should be able to run on any Unix-like platform.

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

To generate the prediction, run

```bash
rm -rf save
mkdir save
python generate.py -c sn_full -e sn_full -t sn
```

## Ensemble

Besides training and evaluation, we want to submit an ensemble of ConvNets to improve performance. These can be done by

```bash
python ensemble.py
```

## Reference

We use a lot of code from [umich-vl/pose-ae-train](https://github.com/umich-vl/pose-ae-train).
