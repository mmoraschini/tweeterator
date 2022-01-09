# tweeterator

## Summary
TWEET genERATOR.

Train a Neural Network (NN) to generate tweets. It is basically a text generator, with some heuristics for twitter data.

The training is done word by word, i.e. the network is trained to generate the next word from the previous ones.

Part-Of-Speech (POS) tags can be used to increase the plausibility of the generated text by modifying the next word probability distribution, based on the predicted next word type and the expected type. POS tags are extracted using `nltk`.

## Usage

The files `tweeterator.py` and `tweeterator_pos.py` can be used both as modules, by loading the files and running the function `train`, or as executables that train the model and save the results to the specified location.
* `tweeterator.py` trains a single NN to predict the next word based on the preceding N words.
* `tweeterator_pos.py` can be used either to train two NN, one that predicts the next word based on the previous N words and one that predicts the next POS tag based on the previous N POS tags, or to train one single NN that predicts the next word and takes the previous N words as one input and the previous N POS tags as the other input. 
The input file must be either a csv or an Excel® file and is read using pandas. The input file must have all tweets in the same column and in different lines.

The files `train.ipynb` and `train_pos.ipynb` contain 4 example use cases:
1. (`train.ipynb`) Train a single NN to generate the next word prediction based only on the previous words.
2. (`train.ipynb`) Train a single NN to generate the next word prediction based on the previous words and then modify the next word probability distribution using the previous POS tags. A routine calculates the probability of seeing a given POS tag based on the previous N POS tags as a simple training data frequency. Each candidate next word is assigned a POS tag and its probability is multiplied by the probability of the POS tag.
3. (`train_pos.ipynb`) Train two NN, use to predict the next word and one to predict the next POS tag. The next POS tag probability is used to modify the next word probability in a similar way to what is done in use case #2, but instead of using the observed frequencies, the second NN output is used.
4. (`train_pos.ipynb`) Train one single NN to predict the next word based on two inputs: the previous N words and the previous N POS tags.

# Warning

Something could break. I still have to write all test cases.

# ToDo

* Add test cases
* Make code work when it sees and unknown word
* Move text generation from ipynb files to tweeterator*.py files
