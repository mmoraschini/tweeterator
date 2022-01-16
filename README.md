# tweeterator

## Summary
TWEET genERATOR.

Contains utilities to train Neural Networks (NN) to generate tweets. It is basically a text generator, with some heuristics for twitter data.

The training is done word by word, i.e. the network is trained to generate the next word from the previous ones.

Part-Of-Speech (POS) tags can be used to increase the plausibility of the generated text by modifying the next word probability distribution, based on the predicted next word type and the expected type. POS tags are extracted using `nltk`.

## Usage

The code is organised in modules:
* `nets` contains the definition of 3 recurrent NN types that can be used to generate text.
  * `one_input_one_output`: wants as input one sequence (either of words or of POS) and outputs the next element prediction.
  * `two_inputs_one_output`: wants as input two sequences, one of words and one of POS, and outputs the next word.
  * `two_inputs_two_outputs`: wants as input two sequences, one of words and one of POS, and outputs the next word and the next POS. The predicted POS can then be used as next input.
* `loader`: contains the data loader, that starts from an input csv or Excel file and returns a list of list, where each element of the inner list contains a word.
* `generators`: contains data generators, both for one source (words or POS) and for both.
* `sentence_generation`: contains routines to generate data for the various net types.
* `text_cleaning`: contains utilities to clean the text.
* `pos_tagging`: contains utilities to add POS tags to sentences and to get their observed frequency.

## Try it

`tweeterator.ipynb` is a Jupyter Notebook the shows how to use the various modules to train and test the different net types.

# Warning

Something could break. I still have to write all test cases.

# ToDo

* Add test cases
