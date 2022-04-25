from typing import List, Optional

import numpy as np
from tensorflow.keras.models import Model
import nltk


def get_prediction(prob: List[float], deterministic: bool) -> int:
    if deterministic:
        chosen_guess = np.argmax(prob)
    else:
        prob = np.array(prob) / np.sum(prob)
        chosen_guess = np.random.choice(range(len(prob)), 1, p=prob)[0]
    
    return chosen_guess


class SentenceGenerator():

    def __init__(self, search_type: str, window: int, w2i: dict, i2w: dict,
                 deterministic: bool, output_length: int, stop_at_eos: bool,
                 pos2i: dict = None, pos_freq: dict = None, min_freq: Optional[float] = None ) -> None:
        """
        Class to generate sentences from various types of models

        Args:
            search_type (str): type of search algorithm for next word prediction, either 'greedy' or 'beam'
            window (int): input sequence length
            w2i (dict): word to int dictionary
            i2w (dict): int to word dictionary
            deterministic (bool): whether to generate always the same sentence or to choose words based on their probability
            output_length (int): length of the output sentence
            stop_at_eos (int): whether to stop when End Of Sentence is generated
            pos2i (dict): POS to int dictionary (if needed, default None)
            pos_freq (dict): frequencies of next POS tag given `window` previous ones (if needed, default None)
            min_freq (float): minimum frequency observed, used to assign a default probability when a key is not found in pos_freq (if needed, default None)
        """
        self.type = search_type
        self.window = window
        self.w2i = w2i
        self.i2w = i2w
        self.determ = deterministic
        self.out_len = output_length
        self.stop_at_eos = stop_at_eos
        self.p2i = pos2i
        self.pos_freq = pos_freq
        self.min_freq = min_freq
        
    def one_model_one_input_one_output(self, model: Model, starting_words: List[str], use_pos_info: bool) -> List[str]:
        """
        Generate a sentence from the `nets.one_input_one_output` model

        Args:
            model (Model): model to generate the sentence from
            starting_words (List[str]): starting words to generate the sentence from
            use_pos_info (bool): use POS tags information when generation. If True pos_freq and min_freq must be provided

        Returns:
            List[str]: list of generated words
        """
        eos_int = self.w2i['.']

        # Set the first window elements to the start of the phrase
        output_int = [self.w2i[word] for word in starting_words]

        # Predict the next word from the preceding ones (using the words already predicted)
        for i in range(0, self.out_len - self.window):
            input_int = np.expand_dims(output_int[i:self.window+i], axis=(0, 2))
            prediction = model(input_int).numpy()[0]
            
            if use_pos_info:
                input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
                words_and_pos_tags = nltk.pos_tag(input_str)
                pos_tags = list(zip(*words_and_pos_tags))[1]
                preceding = str(pos_tags)
                try:
                    pos_freqs = self.pos_freq[preceding]
                except KeyError:
                    pos_freqs = None
                
                best_guesses = np.argsort(prediction)[::-1][:10]
                posterior = []
                for guess in best_guesses:
                    # Get the POS tag for the examined word
                    test_str = input_str + [self.i2w[guess]]

                    guess_pos = nltk.pos_tag(test_str)[-1][1]
                    try:
                        pos_prob = pos_freqs[guess_pos]
                    except KeyError:
                        # If this sequence was never observed give it a probability equal to the minimum frequence observed
                        pos_prob = self.min_freq
                    except TypeError:
                        # if pos_freqs is None because the sequence was never observed, do not vary the probabilities
                        pos_prob = 1

                    prior = prediction[guess]

                    posterior.append(prior * pos_prob)
                
                chosen_guess = get_prediction(posterior, self.determ)
                
                word_int = best_guesses[chosen_guess]
            else:
                if self.determ:
                    word_int = np.argmax(prediction)
                else:
                    word_int = np.random.choice(range(len(prediction)), 1, p=prediction)[0]

            output_int.append(word_int)

            if self.stop_at_eos & (word_int == eos_int):
                break
        
        # Convert integers to words
        return [self.i2w[ii] for ii in output_int]


    def one_model_two_inputs_one_output(self, model: Model, starting_words: List[str]) -> List[str]:
        """
        Generate a sentence from the `nets.two_inputs_one_output` model

        Args:
            model (Model): model to generate the sentence from
            starting_words (List[str]): starting words to generate the sentence from

        Returns:
            List[str]: list of generated words
        """
        eos_int = self.w2i['.']

        # Set the first window elements to the start of the phrase
        output_int = [self.w2i[word] for word in starting_words]

        ###############
        # Beam search #
        ###############
        nbeams = 5
        solutions = [output_int]
        scores = [0]

        max_i = self.out_len - self.window

        for i in range(max_i):
            new_solutions = []
            new_scores = []
            for j in range(len(solutions)):
                sln = solutions[j]
                scr = scores[j]

                input_int_word = np.expand_dims(sln[i:self.window+i], axis=(0, 2))

                input_str = [self.i2w[ii] for ii in sln[i:self.window+i]]
                words_and_pos_tags = nltk.pos_tag(input_str)
                pos_tags = list(zip(*words_and_pos_tags))[1]
                pos_tags_int = [self.pos2i[pos] for pos in pos_tags]
                pos_tags_int = np.expand_dims(pos_tags_int, axis=(0, 2))
                
                input_str = [self.i2w[ii] for ii in sln[i:self.window+i]]
                
                predictions = model([input_int_word, pos_tags_int]).numpy()[0]

                idx = np.argsort(predictions)[-nbeams:]
                for ii in idx:
                    new_sln = sln + [ii]
                    new_scr = scr - np.log(predictions[ii])

                    new_solutions.append(new_sln)
                    new_scores.append(new_scr)
            
            idx = np.argsort(new_scr)[-nbeams:]
            solutions = [new_solutions[ii] for ii in idx]
            scores = [new_scores[ii] for ii in idx]


        ###############
        #   Greedy    #
        ###############

        # Predict the next word from the preceding ones (using the words already predicted)
        for i in range(0, self.out_len - self.window):
            input_int_word = np.expand_dims(output_int[i:self.window+i], axis=(0, 2))

            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            words_and_pos_tags = nltk.pos_tag(input_str)
            pos_tags = list(zip(*words_and_pos_tags))[1]
            pos_tags_int = [self.pos2i[pos] for pos in pos_tags]
            pos_tags_int = np.expand_dims(pos_tags_int, axis=(0, 2))
            
            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            
            prediction = model([input_int_word, pos_tags_int]).numpy()[0]

            word_int = get_prediction(prediction, self.determ)

            output_int.append(word_int)

            if self.stop_at_eos & (word_int == eos_int):
                break

        # Convert integers to words
        return [self.i2w[ii] for ii in output_int]


    def one_model_two_inputs_two_outputs(self, model: Model, starting_words: List[str]) -> List[str]:
        """
        Generate a sentence from the `nets.two_inputs_two_outputs` model

        Args:
            model (Model): model to generate the sentence from
            starting_words (List[str]): starting words to generate the sentence from

        Returns:
            List[str]: list of generated words
        """
        eos_int = self.w2i['.']

        # Set the first window elements to the start of the phrase
        output_int = [self.w2i[word] for word in starting_words]

        # Predict the next word from the preceding ones (using the words already predicted)
        for i in range(0, self.out_len - self.window):
            input_int_word = np.expand_dims(output_int[i:self.window+i], axis=(0, 2))

            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            words_and_pos_tags = nltk.pos_tag(input_str)
            pos_tags = list(zip(*words_and_pos_tags))[1]
            pos_tags_int = [self.pos2i[pos] for pos in pos_tags]
            pos_tags_int = np.expand_dims(pos_tags_int, axis=(0, 2))
            
            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            
            prediction, prediction_pos = model([input_int_word, pos_tags_int])
            prediction = prediction.numpy()[0]
            prediction_pos = prediction_pos.numpy()[0]

            best_guesses = np.argsort(prediction)[::-1][:10]
            posterior = []
            for guess in best_guesses:
                # Get the POS tag for the examined word
                test_str = input_str + [self.i2w[guess]]
                guess_pos = nltk.pos_tag(test_str)[-1][1]

                prior = prediction[guess]

                # Get index of POS in prediction output
                pos_i = self.pos2i[guess_pos]
                pos_prob = prediction_pos[pos_i]
                posterior.append(prior * pos_prob)

            chosen_guess = get_prediction(posterior, self.determ)
            
            word_int = best_guesses[chosen_guess]

            output_int.append(word_int)

            if self.stop_at_eos & (word_int == eos_int):
                break
        
        # Convert integers to words
        return [self.i2w[ii] for ii in output_int]


    def two_models(self, model: List[Model], starting_words: List[str]) -> List[str]:
        """
        Generate a sentence from two `nets.one_input_one_output` models. One that predicts the next word and the other the next POS.

        Args:
            model (List[Model]): models to generate the sentence from
            starting_words (List[str]): starting words to generate the sentence from

        Returns:
            List[str]: list of generated words
        """

        eos_int = self.w2i['.']

        # Set the first window elements to the start of the phrase
        output_int = [self.w2i[word] for word in starting_words]

        # Predict the next word from the preceding ones (using the words already predicted)
        for i in range(0, self.out_len - self.window):
            input_int_word = np.expand_dims(output_int[i:self.window+i], axis=(0, 2))

            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            words_and_pos_tags = nltk.pos_tag(input_str)
            pos_tags = list(zip(*words_and_pos_tags))[1]
            pos_tags_int = [self.pos2i[pos] for pos in pos_tags]
            pos_tags_int = np.expand_dims(pos_tags_int, axis=(0, 2))
            
            input_str = [self.i2w[ii] for ii in output_int[i:self.window+i]]
            
            prediction = model[0](input_int_word).numpy()[0]
            prediction_pos = model[1](pos_tags_int).numpy()[0]

            best_guesses = np.argsort(prediction)[::-1][:10]
            posterior = []
            for guess in best_guesses:
                # Get the POS tag for the examined word
                test_str = input_str + [self.i2w[guess]]
                guess_pos = nltk.pos_tag(test_str)[-1][1]

                prior = prediction[guess]

                # Get index of POS in prediction output
                pos_i = self.pos2i[guess_pos]
                pos_prob = prediction_pos[pos_i]
                posterior.append(prior * pos_prob)

            chosen_guess = get_prediction(posterior, self.determ)
            
            word_int = best_guesses[chosen_guess]

            output_int.append(word_int)

            if self.stop_at_eos & (word_int == eos_int):
                break

        # Convert integers to words
        return [self.i2w[ii] for ii in output_int]
