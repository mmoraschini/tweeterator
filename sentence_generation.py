from typing import List

import numpy as np
from tensorflow.keras.models import Model
import nltk


def one_model_one_input_one_output(model: Model, starting_words: List[str], window: int, use_pos_info: bool,
                                w2i: dict, i2w: dict, pos_freq: dict, min_freq: float,
                                deterministic: bool, output_length: int) -> List[str]:
    """
    Generate a sentence from the `nets.one_input_one_output` model

    Args:
        model (Model): model to generate the sentence from
        starting_words (List[str]): starting words to generate the sentence from
        window (int): input sequence length
        use_pos_info (bool): use POS tags information when generation. If defined pos_freq and min_freq must be provided
        w2i (dict): word to int dictionary
        i2w (dict): int to word dictionary
        pos_freq (dict): frequencies of next POS tag given `window` previous ones 
        min_freq (float): minimum frequency observed, used to assign a default probability when a key is not found in pos_freq
        deterministic (bool): whether to generate always the same sentence or to choose words based on their probability
        output_length (int): length of the output sentence

    Returns:
        List[str]: list of generated words
    """
    output_int = np.empty(output_length, dtype=int)

    # Set the first window elements to the start of the phrase
    output_int[:window] = [w2i[word] for word in starting_words]

    # Predict the next word from the preceding ones (using the words already predicted)
    for i in range(0, output_length - window):
        input_int = output_int[np.newaxis, i:window+i, np.newaxis]
        prediction = model(input_int).numpy()[0]
        
        if use_pos_info:
            # Even if the whole sentence is known in this test, run POS tagging only on the part
            # of sentence preceding the word to generate (real-like scenario)
            input_str = [i2w[ii] for ii in output_int[i:window+i]]
            words_and_pos_tags = nltk.pos_tag(input_str)
            pos_tags = list(zip(*words_and_pos_tags))[1]
            preceding = str(pos_tags)
            try:
                pos_freqs = pos_freq[preceding]
            except KeyError:
                pos_freqs = None
            
            best_guesses = np.argsort(prediction)[::-1][:10]
            posterior = []
            for guess in best_guesses:
                # Get the POS tag for the examined word
                test_str = input_str + [i2w[guess]]

                guess_pos = nltk.pos_tag(test_str)[-1][1]
                try:
                    pos_prob = pos_freqs[guess_pos]
                except KeyError:
                    # If this sequence was never observed give it a probability equal to the minimum frequence observed
                    pos_prob = min_freq
                except TypeError:
                    # if pos_freqs is None because the sequence was never observed, do not vary the probabilities
                    pos_prob = 1

                prior = prediction[guess]

                posterior.append(prior * pos_prob)
        
            if deterministic:
                chosen_guess = np.argmax(posterior)
            else:
                posterior = np.array(posterior) / np.sum(posterior)
                chosen_guess = np.random.choice(range(len(posterior)), 1, p=posterior)[0]
            
            word_int = best_guesses[chosen_guess]
        else:
            if deterministic:
                word_int = np.argmax(prediction)
            else:
                word_int = np.random.choice(range(len(prediction)), 1, p=prediction)[0]

        output_int[window + i] = word_int
    
    # Convert integers to words
    output = []
    for i in range(len(output_int)):
        word_int = output_int[i]
        word = i2w[word_int]
        output.append(word)
    
    return [i2w[ii] for ii in output_int]


def one_model_two_inputs_one_output(model: Model, starting_words: List[str], window: int, w2i: dict, i2w: dict,
                                pos2i: dict, deterministic: bool, output_length: int) -> List[str]:
    """
    Generate a sentence from the `nets.two_inputs_one_output` model

    Args:
        model (Model): model to generate the sentence from
        starting_words (List[str]): starting words to generate the sentence from
        window (int): input sequence length
        w2i (dict): word to int dictionary
        i2w (dict): int to word dictionary
        pos2i (dict): POS to int dictionary
        deterministic (bool): whether to generate always the same sentence or to choose words based on their probability
        output_length (int): length of the output sentence

    Returns:
        List[str]: list of generated words
    """
    output_int_word = np.empty(output_length, dtype=int)

    # Set the first window elements to the start of the phrase
    output_int_word[:window] = [w2i[word] for word in starting_words]

    # Predict the next word from the preceding ones (using the words already predicted)
    for i in range(0, output_int_word.size - window):
        input_int_word = output_int_word[np.newaxis, i:window+i, np.newaxis]

        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        words_and_pos_tags = nltk.pos_tag(input_str)
        pos_tags = list(zip(*words_and_pos_tags))[1]
        pos_tags_int = np.array([pos2i[pos] for pos in pos_tags])
        pos_tags_int = pos_tags_int[np.newaxis, :, np.newaxis]
        
        # Even if the whole sentence is known in this test, run POS tagging only on the part
        # of sentence preceding the word to generate (real-like scenario)
        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        
        prediction = model([input_int_word, pos_tags_int]).numpy()[0]
        if deterministic:
            word_int = np.argmax(prediction)
        else:
            word_int = np.random.choice(range(len(prediction)), 1, p=prediction)[0]

        output_int_word[window + i] = word_int

    # Convert integers to words
    output = []
    for i in range(len(output_int_word)):
        word_int = output_int_word[i]
        word = i2w[word_int]
        output.append(word)
    
    return [i2w[ii] for ii in output_int_word]


def one_model_two_inputs_two_outputs(model: Model, starting_words: List[str], window: int, w2i: dict, i2w: dict,
            pos2i: dict, deterministic: bool, output_length: int) -> List[str]:
    """
    Generate a sentence from the `nets.two_inputs_two_outputs` model

    Args:
        model (Model): model to generate the sentence from
        starting_words (List[str]): starting words to generate the sentence from
        window (int): input sequence length
        w2i (dict): word to int dictionary
        i2w (dict): int to word dictionary
        pos2i (dict): POS to int dictionary
        deterministic (bool): whether to generate always the same sentence or to choose words based on their probability
        output_length (int): length of the output sentence

    Returns:
        List[str]: list of generated words
    """
    output_int_word = np.empty(output_length, dtype=int)

    # Set the first window elements to the start of the phrase
    output_int_word[:window] = [w2i[word] for word in starting_words]

    # Predict the next word from the preceding ones (using the words already predicted)
    for i in range(0, output_int_word.size - window):
        input_int_word = output_int_word[np.newaxis, i:window+i, np.newaxis]

        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        words_and_pos_tags = nltk.pos_tag(input_str)
        pos_tags = list(zip(*words_and_pos_tags))[1]
        pos_tags_int = np.array([pos2i[pos] for pos in pos_tags])
        pos_tags_int = pos_tags_int[np.newaxis, :, np.newaxis]
        
        # Even if the whole sentence is known in this test, run POS tagging only on the part
        # of sentence preceding the word to generate (real-like scenario)
        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        
        prediction, prediction_pos = model([input_int_word, pos_tags_int])
        prediction = prediction.numpy()[0]
        prediction_pos = prediction_pos.numpy()[0]

        best_guesses = np.argsort(prediction)[::-1][:10]
        posterior = []
        for guess in best_guesses:
            # Get the POS tag for the examined word
            test_str = input_str + [i2w[guess]]
            guess_pos = nltk.pos_tag(test_str)[-1][1]

            prior = prediction[guess]

            # Get index of POS in prediction output
            pos_i = pos2i[guess_pos]
            pos_prob = prediction_pos[pos_i]
            posterior.append(prior * pos_prob)

        if deterministic:
            chosen_guess = np.argmax(posterior)
        else:
            posterior = np.array(posterior) / np.sum(posterior)
            chosen_guess = np.random.choice(range(len(posterior)), 1, p=posterior)[0]
        
        word_int = best_guesses[chosen_guess]

        output_int_word[window + i] = word_int

    # Convert integers to words
    output = []
    for i in range(len(output_int_word)):
        word_int = output_int_word[i]
        word = i2w[word_int]
        output.append(word)
    
    return [i2w[ii] for ii in output_int_word]


def two_models(model: List[Model], starting_words: List[str], window: int, w2i: dict, i2w: dict,
            pos2i: dict, deterministic: bool, output_length: int) -> List[str]:
    """
    Generate a sentence from two `nets.one_input_one_output` models. One that predicts the next word and the other the next POS.

    Args:
        model (List[Model]): models to generate the sentence from
        starting_words (List[str]): starting words to generate the sentence from
        window (int): input sequence length
        w2i (dict): word to int dictionary
        i2w (dict): int to word dictionary
        pos2i (dict): POS to int dictionary
        deterministic (bool): whether to generate always the same sentence or to choose words based on their probability
        output_length (int): length of the output sentence

    Returns:
        List[str]: list of generated words
    """
    
    output_int_word = np.empty(output_length, dtype=int)

    # Set the first window elements to the start of the phrase
    output_int_word[:window] = [w2i[word] for word in starting_words]

    # Predict the next word from the preceding ones (using the words already predicted)
    for i in range(0, output_int_word.size - window):
        input_int_word = output_int_word[np.newaxis, i:window+i, np.newaxis]

        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        words_and_pos_tags = nltk.pos_tag(input_str)
        pos_tags = list(zip(*words_and_pos_tags))[1]
        pos_tags_int = np.array([pos2i[pos] for pos in pos_tags])
        pos_tags_int = pos_tags_int[np.newaxis, :, np.newaxis]
        
        # Even if the whole sentence is known in this test, run POS tagging only on the part
        # of sentence preceding the word to generate (real-like scenario)
        input_str = [i2w[ii] for ii in output_int_word[i:window+i]]
        
        prediction = model[0](input_int_word).numpy()[0]
        prediction_pos = model[1](pos_tags_int).numpy()[0]

        best_guesses = np.argsort(prediction)[::-1][:10]
        posterior = []
        for guess in best_guesses:
            # Get the POS tag for the examined word
            test_str = input_str + [i2w[guess]]
            guess_pos = nltk.pos_tag(test_str)[-1][1]

            prior = prediction[guess]

            # Get index of POS in prediction output
            pos_i = pos2i[guess_pos]
            pos_prob = prediction_pos[pos_i]
            posterior.append(prior * pos_prob)

        if deterministic:
            chosen_guess = np.argmax(posterior)
        else:
            posterior = np.array(posterior) / np.sum(posterior)
            chosen_guess = np.random.choice(range(len(posterior)), 1, p=posterior)[0]
        
        word_int = best_guesses[chosen_guess]

        output_int_word[window + i] = word_int

    # Convert integers to words
    output = []
    for i in range(len(output_int_word)):
        word_int = output_int_word[i]
        word = i2w[word_int]
        output.append(word)
    
    return [i2w[ii] for ii in output_int_word]
