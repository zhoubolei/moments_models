# Pretrained models for Moments in Time Dataset

We release the pre-trained models trained on [Moments in Time](http://moments.csail.mit.edu/).

### Download the Models

* Clone the code from Github:
```
    git clone https://github.com/metalbubble/moments_models.git
    cd moments_models
```

### Models

* RGB model in PyTorch (ResNet50 pretrained on ImageNet). Tested sucessfully in PyTorch0.3 + python36.
```
    python test_model.py
```

* Dynamic Image model in Caffe (TODO).

* TRN models is already at [this repo](https://github.com/metalbubble/TRN-pytorch). Please go there to download the models.


### Reference

Mathew Monfort, Bolei Zhou, Sarah Adel Bargal, Alex Andonian, Tom Yan, Kandan Ramakrishnan, Lisa Brown, Quanfu Fan, Dan Gutfreund, Carl Vondrick, Aude Oliva. 'Moments in Time Dataset: one million videos for event understanding'. arXiv. XXX:XXX. 


### Acknowledgements

The project is supported by MIT-IBM Watson AI Lab and IBM Research.
