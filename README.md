# tweeterator

## Summary
TWEET genERATOR.

Train a Neural Network to generate tweets. It is basically a text generator, with some heuristics for twitter data.

The training is done word by word, i.e. the network is trained to generate the next word from the preceding ones.

## Usage

The file `tweeterator.py` can be used both as a module containing the function `train` or as an executable file that trains the model and saves the result to the specified location

## Try the library

The file `train.ipynb` contains an example use case. The input file must be either a csv or an excel file and is read using pandas. For the code to work each tweet must be in a separate line and the name of the column containing the tweets must be provided.

`train.ipynb` contains also a Part-Of-Speech (POS) tagging routine that helps in determining the best word to generate, based on the order of the synctactic elements preceding the word to generate. The probabilities are assigned by computing the frequencies on the training data.
