## Basic Tensorflow Model Training and Saving Code

### [Tensor.py](tensor.py) 
This File Uses the Fashion MNIST dataset and trains a sequential model. This also has some basic Checkpoint system from Keras callback api. The code saves the model as `model.h5`

```python

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

```
The script also logs the results of the compiled model which can be viewed on tensorboard using :
```
tensorboard --logdir logs/fit 
```

### [Tensor2.py](tensor2.py)

This file uses subclassing system of keras in order to create a simple model on MNIST dataset. It also includes a model weights saving system for now as we can't really use `model.save()` in subclassing method.

The [mode2.h5](mode2.h5) is the saved numpy array for models. The loading system doesn't seem to be working properly atm ☹.