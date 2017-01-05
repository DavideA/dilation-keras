# Multi-Scale Context Aggregation by Dilated Convolutions in Keras
This repository holds a Keras porting of the [ICLR 2016 paper](https://arxiv.org/abs/1511.07122) by Yu and Koltun. 
It holds the four semantic segmentation pretrained networks that you can find in the [original repo](https://github.com/fyu/dilation) (Caffe).

##How to use
To use the pretrained models, you need to get the h5 files that store parameters. Simply follow the instructions in `weights/download_weights.txt`.

Please note that the porting works on with the Theano backend. To use it with Tensorflow, you'll have to convert model weights with [convert kernel](https://keras.io/utils/np_utils/#convert_kernel).

Once you have the weights, just run the predict.py script
* the models are in `models.py`
* `predict.py` calls a prediction on a test image, using the same routine that was released for the Caffe version.

**Cityscapes model disclaimer:** I didn't manage to convert the final upsampling layer (deconv with grouping), so I replaced it with Upsampling + Convolution.
