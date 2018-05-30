# Adversarial Constraint Learning
Source code for our IJCAI paper ["Adversarial Constraint Learning for Structured Prediction"](https://arxiv.org/abs/1805.10561).

If you find it helpful, please consider citing our paper.

    @article{ren2018adversarial,
      title={Adversarial Constraint Learning for Structured Prediction},
      author={Ren, Hongyu and Stewart, Russell and Song, Jiaming and Kuleshov, Volodymmyr and Ermon, Stefano},
      journal={arXiv preprint arXiv:1805.10561},
      year={2018}
    }

## Requirements
1. python 2.7

2. Tensorflow 1.6.0


## Training
```
python acl.py --hypes hypes/double_v2.json --num_with_label 5 --iters 20000 --logdir path/to/log
```

## Samples

<img src="https://github.com/hyren/acl/blob/master/images/pose-1.gif"><img src="https://github.com/hyren/acl/blob/master/images/pose-2.gif"><img src="https://github.com/hyren/acl/blob/master/images/pose-3.gif"><img src="https://github.com/hyren/acl/blob/master/images/pose-4.gif">

If you have any questions, feel free to contact <hyren@cs.stanford.edu>.
