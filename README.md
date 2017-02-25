# Multi-Scale Context Aggregation by Dilated Convolutions in Keras
This repository holds a Keras porting of the [ICLR 2016 paper](https://arxiv.org/abs/1511.07122) by Yu and Koltun. 
It holds the four semantic segmentation pretrained networks that you can find in the [original repo](https://github.com/fyu/dilation) (Caffe).

## How to use
Just use the `DilationNet` function in `dilation_net.py` to get the model.
To see an example, run `predict.py`.

Please note that the porting works on with the Theano dim ordering.
Tensorflow backend should since if needed, the function `convert_all_kernels_in_model` is called.
However, it is not tested.

**Cityscapes model disclaimer:** I didn't manage to convert the final upsampling layer (deconv with grouping), so I replaced it with Upsampling + Convolution.
