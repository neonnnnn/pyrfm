# pyrfm
A library for random feature maps and linear models with random feature maps in Python.

[![Build Status](https://travis-ci.org/neonnnnn/pyrfm.svg?branch=master)](https://travis-ci.org/neonnnnn/pyrfm)

## Installation
 1. Download the source codes by


    git clone https://github.com/neonnnnn/pyrfm.git

  or download as a ZIP from GitHub.

 2. Install the dependencies:


    cd pyrfm

    pip install -r requirements.txt

 3. Finally, build and install pyrfm by


    python setup.py install

For running example codes (pyrfm/benchmarks), jupyter and matplotlib are required.

## Documentation
https://neonnnnn.github.io/pyrfm/

## What are random feature maps?
Using random feature maps is a promising way for large-scale kernel methods.
They are maps from an original feature space to a randomized feature space 
approximating a kernel-induced feature space.
The idea is to run linear models on such randomized feature space for 
classification, regression, clustering, etc.
When the dimension of the random feature map D is not so high and the number of
training example N is large, this approach is very efficient compared to 
canonical kernel methods.

## Authors
 - Kyohei Atarashi, 2018-present
