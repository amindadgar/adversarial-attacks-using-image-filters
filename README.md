# adversarial-attacks-using-image-filters
This repository is an implementation of https://link.springer.com/chapter/10.1007/978-3-030-72699-7_35 article. It uses evolutionary strategy to configure image filter parameters in order to attack adversarially to a neural network.

For image filters, we've been using the implementations from https://github.com/ruvvet/opencv_filters and gingham from the library [pilgram](https://pypi.org/project/pilgram/). There were some modifications for image filters to configure the parameters mentioned in the article. Also a method named feature squeezing is being used which the code is from the owner of it in https://github.com/mzweilin/EvadeML-Zoo/blob/master/Reproduce_FeatureSqueezing.md 

## Steps to run the project:
- `pip install -r requirements.txt`
- `mkdir results`
- either run one of these, with change of configuaration in them (if you want) 
    - `python main.py`
    - `python main.ipynb` 

*Note:* This project is using just the 100 first images in training set of CIFAR-10 because of low computational resources available, but you can increase the images using the `image_portion` initial variable in `fitness.py`. 
