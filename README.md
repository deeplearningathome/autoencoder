# Constrained de-noising AutoEncoder
This is (an intentionally) simple implementation of constrained de-noising auto-encoder.
And auto-encoder is an *unsupervised* learning model, which takes some input, runs it though "encoder" 
part to get *encodings* of the input. Then it attempts to reconstruct original input based only on obtained encodings.

More details and explanations can be found here: [DeepLearning@Home](http://www.deeplearningathome.com) 

The idea is that *encodings* will encode the most important information in the data.

|File | Description |
------|-----|
autoencoder.py | contains AutoEncoder class. Is not application specific, multilayer architecutre.
autoencider_use_mnist.py | Example of using the AutoEncoder for MNIST image compression/representation
utils.py | different helper functions
-----------------------------------

Example run:
```python
python ~/repos/autoencoder/autoencoder_use_mnist.py --encoder_network=784,128,10 --noise_level=0.0 --batch_size=64 --num_epochs=60 --logdir=LD_784_128_10_N0
```
Then start Tensorboard:
```
tensorboard --logdir=LD_784_128_10_N0
```