# Applying inverse graphics to image tracing using unsupervised generative learning

### Setup

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Testing
Make sure to add `renderer.pkl` and `actor.pkl` before testing.
The Jupyter notebooks were used to generate the results and are not guarenteed to work.

## Training

### Datasets
- Iconator (https://www.vectornator.io/icons)
- PublicDomainVector (https://publicdomainvectors.org)
- Logos (https://github.com/gilbarbara/logos)

### Neural Renderer
To create a differentiable painting environment, we need train the neural renderer firstly. 

```
$ python3 baseline/train_renderer.py
$ tensorboard --logdir train_log --port=6006
(The training process will be shown at http://127.0.0.1:6006)
```

### Paint Agent
After the neural renderer looks good enough, we can begin training the agent.
```
$ cd baseline
$ python3 train.py --max_step=40 --debug --batch_size=96
(A step contains 5 strokes in default.)
$ tensorboard --logdir train_log --port=6006
```
