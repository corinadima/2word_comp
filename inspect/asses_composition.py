#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import operator
import math
import itertools
import random
from pathlib import Path

from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors


def read_ranks(fileName):
    dictionary = {}
    with open(fileName, mode='rt', encoding='utf8') as inf:
        for line in inf:
            splits = line.strip().split(' ')
            assert(len(splits) == 2), "error: wrong line %s" % line
            word = splits[0]
            rank = int(splits[1])
            dictionary[word] = rank
    return dictionary

def compute_neighbours(chosen, path_composed_emb, path_observed_emb, neighbours):
    """
        Returns the neighbours of words from a composed space in an observed space.
    """
    nearest_words = {}
    nearest_cosines = {}

    raw_observed_space = Word2VecKeyedVectors.load_word2vec_format(path_observed_emb, binary=False)
    norm_observed_space = normalize(raw_observed_space.vectors, norm="l2", axis=1)

    raw_composed_space = Word2VecKeyedVectors.load_word2vec_format(path_composed_emb, binary=False)
    norm_composed_space = normalize(raw_composed_space.vectors, norm="l2", axis=1)

    chosen_words = set([tup[0] for tup in chosen])

    composed_words = raw_composed_space.wv.vocab
    observed_words = raw_observed_space.wv.vocab

    for w_idx, word in enumerate(composed_words):
        if (word not in chosen_words): continue
        vector = composedSpace.get_row(word)
        Y = 1 - cdist(vector.mat, observedSpace.get_cooccurrence_matrix().mat, 'cosine')
        shape = Y.shape
        Yr = Y.reshape(shape[1])

        # print(Yr.shape)
        nearest_k_indices = np.argpartition(Yr, -neighbours)[-neighbours:]
        sorted_nearest_k_indices = nearest_k_indices[np.argsort(Yr[nearest_k_indices])]

        neighbour_words = [observed_words[idx] for idx in reversed(sorted_nearest_k_indices)]
        neighbour_cosines = [Yr[idx] for idx in reversed(sorted_nearest_k_indices)]

        nearest_words[word] = neighbour_words
        nearest_cosines[word] = neighbour_cosines

    return nearest_words, nearest_cosines

def compute_best_rank_average(best_rank, dev_ranks, path_composed_emb, path_observed_emb):
    chosen_words = set([tup for tup in dev_ranks if tup[1] == best_rank])

    nearest_words, nearest_cosines = compute_neighbours(chosen_words, path_composed_emb, path_observed_emb, 5)
    rank_cosines = [cosine[best_rank - 1] for word, cosine in nearest_cosines.items()]
    print(rank_cosines)
    avg_cosine = np.mean(rank_cosines)
    print(str(len(chosen_words)) + " words of rank " + str(best_rank))
    print(" average cosine " + str(avg_cosine))


def latex_print_info(output_file, sample_size, tuple_list, head_dev_ranks, nearest_words, nearest_cosines, neighbours):
    with open(output_file, mode='w', encoding='utf8') as out:
        out.write("\n")

        out.write("\\documentclass{article}\n") 
        out.write("\\usepackage[utf8]{inputenc}\n") 
        out.write("\\usepackage{booktabs}\n")   
        out.write("\\begin{document}\n")
        out.write("\\begin{table}[!tbh]\n")
        out.write("\\begin{center}\n")
        out.write("\\scriptsize\n")
        out.write("\\begin{tabular}{rrrr}\n")

        for i in range(math.floor(len(tuple_list)/sample_size)):
            format_string = "\\textbf{%s:%d} & " * (sample_size-1) + "\\textbf{%s:%d} \\\\\n "
            tp = tuple_list[i*sample_size:(i+1)*sample_size]
            ctp = list(itertools.chain(*tp))
            out.write(format_string % tuple(ctp[:]))
            out.write("\n\midrule\n")

            for j in range(neighbours):     
                format_string = "%s %.2f & " * (sample_size-1) + "%s %.2f \\\\\n "
                nw = [(nearest_words[tup[0]][j]) for tup in tp]
                nc = [nearest_cosines[tup[0]][j] for tup in tp]
                ncw = zip(nw, nc)
                out.write(format_string % tuple(list(itertools.chain(*ncw))))
            out.write("\n\midrule")

        out.write("\end{tabular}\n")
        out.write("\caption{\label{ch5:table:de_nn-only_dev_head_examples}}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")
        out.write("\end{document}\n")

def select_sample(sample_size, start_rank, ranks_dict):
    samples = []
    sorted_keys = sorted(ranks_dict)
    current_list = random.sample(ranks_dict[start_rank], len(ranks_dict[start_rank]))
    current_key_idx = sorted_keys.index(start_rank)
    # ipdb.set_trace()
    while (len(samples) < sample_size):
        missing = sample_size - len(samples)
        samples = samples + current_list[:missing]

        if (current_key_idx < len(sorted_keys)-1):
            current_key_idx += 1
            current_list = random.sample(ranks_dict[sorted_keys[current_key_idx]], len(ranks_dict[sorted_keys[current_key_idx]]))

    return samples


def asses_composition(path_observed_emb, path_composed_emb, path_ranks, model_name):
    print(model_name)
    comp_test_ranks = read_ranks(path_ranks)
    sorted_test = sorted(comp_test_ranks.items(), key=operator.itemgetter(1), reverse=False)

    test_ranks_dict = {tup[1]:[t for t in sorted_test if t[1]==tup[1]] for tup in sorted_test}

    sample_size = 4
    best_rank = sorted_test[0][1]
    high_rank_list = select_sample(sample_size, best_rank, test_ranks_dict)
    print(high_rank_list)

    middle_rank = sorted_test[math.floor(len(sorted_test)/2)-int(math.floor(sample_size/2))][1]
    middle_rank_list = select_sample(sample_size, middle_rank, test_ranks_dict)
    print(middle_rank_list)

    lowest_rank = sorted_test[-1][1]
    low_rank_list = select_sample(sample_size, lowest_rank, test_ranks_dict) 
    print(low_rank_list)

    all_ranks = list(itertools.chain(high_rank_list, middle_rank_list, low_rank_list))

    neighbours = 5
    nearest_words, nearest_cosines = compute_neighbours(all_ranks, path_composed_emb, path_observed_emb, neighbours)

    compute_best_rank_average(best_rank, sorted_test, path_composed_emb, path_observed_emb)

    output_file = str(Path('data/results/German/' + model_name + '_analysis.tex'))
    latex_print_info(output_file, sample_size, all_ranks, comp_test_ranks, nearest_words, nearest_cosines, neighbours)

if __name__=="__main__":

    path_observed_embeddings = str(Path('data/german_compounds_nn_only_composition_dataset/embeddings/glove_decow14ax_all_min_100_vectors_raw/glove_decow14ax_all_min_100_vectors_raw.300d_cmh.dm'))

    head_ranks_file = str(Path('data/results/German/model_HeadOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test_rankedCompounds.txt'))
    head_predictions_file = str(Path('data/results/German/model_HeadOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test.pred'))

    modifier_ranks_file = './data/german_compounds_nn_only_composition_dataset/test_learned_representations/model_ModifierOnly_tanh_adagrad_batch100_mse_2017-07-13_14-51_dev_rankedCompounds.txt'
    modifier_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_ModifierOnly_tanh_adagrad_batch100_mse_2017-07-13_14-51_dev.pred'

    addition_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Addition_tanh_adagrad_batch100_mse_2017-07-13_15-01_dev_rankedCompounds.txt'
    addition_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Addition_tanh_adagrad_batch100_mse_2017-07-13_15-01_dev.pred'

    mul_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Multiplication_tanh_adagrad_batch100_mse_2017-07-13_15-02_dev_rankedCompounds.txt'
    mul_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Multiplication_tanh_adagrad_batch100_mse_2017-07-13_15-02_dev.pred'

    w_addition_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_WeightedAddition_tanh_adagrad_batch100_mse_2017-07-13_15-11_dev_rankedCompounds.txt'
    w_addition_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_WeightedAddition_tanh_adagrad_batch100_mse_2017-07-13_15-11_dev.pred'

    lexfunc_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_LexicalFunction_tanh_adagrad_batch100_mse_2017-07-13_14-23_dev_rankedCompounds.txt'
    lexfunc_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_LexicalFunction_tanh_adagrad_batch100_mse_2017-07-13_14-23_dev.pred'

    fulladd_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_FullAdd_tanh_adagrad_batch100_mse_2017-07-13_15-11_dev_rankedCompounds.txt'
    fulladd_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_FullAdd_tanh_adagrad_batch100_mse_2017-07-13_15-11_dev.pred'

    dil_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Dilation_tanh_adagrad_batch100_mse_2017-07-13_15-20_dev_rankedCompounds.txt'
    dil_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Dilation_tanh_adagrad_batch100_mse_2017-07-13_15-20_dev.pred'

    matrix_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Matrix_tanh_adagrad_batch100_mse_2017-07-13_15-31_dev_rankedCompounds.txt'
    matrix_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_Matrix_tanh_adagrad_batch100_mse_2017-07-13_15-31_dev.pred'

    fulllex_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_FullLex_tanh_adagrad_batch100_mse_2017-07-26_15-59_dev_rankedCompounds.txt'
    fulllex_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_FullLex_tanh_adagrad_batch100_mse_2017-07-26_15-59_dev.pred'

    addmask_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_AddMask_tanh_adagrad_batch100_mse_2017-07-20_12-10_dev_rankedCompounds.txt'
    addmask_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_AddMask_tanh_adagrad_batch100_mse_2017-07-20_12-10_dev.pred'

    wmask_ranks_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_WMask_tanh_adagrad_batch100_mse_2017-07-20_11-29_dev_rankedCompounds.txt'
    wmask_predictions_file = './data/german_compounds_nn_only_composition_dataset/dev_learned_representations/model_WMask_tanh_adagrad_batch100_mse_2017-07-20_11-29_dev.pred'


    asses_composition(path_observed_embeddings, head_predictions_file, head_ranks_file, 'head_only')
    # asses_composition(path_observed_embeddings, modifier_predictions_file, modifier_ranks_file, 'modifier_only')
    # asses_composition(path_observed_embeddings, addition_predictions_file, addition_ranks_file, 'addition')
    # asses_composition(path_observed_embeddings, mul_predictions_file, mul_ranks_file, 'mul')

    # asses_composition(path_observed_embeddings, w_addition_predictions_file, w_addition_ranks_file, 'w_addition')
    # asses_composition(path_observed_embeddings, lexfunc_predictions_file, lexfunc_ranks_file, 'lexfunc')
    # asses_composition(path_observed_embeddings, fulladd_predictions_file, fulladd_ranks_file, 'fulladd')
    # asses_composition(path_observed_embeddings, dil_predictions_file, dil_ranks_file, 'dil')
    # asses_composition(path_observed_embeddings, matrix_predictions_file, matrix_ranks_file, 'matrix')
    # asses_composition(path_observed_embeddings, fulllex_predictions_file, fulllex_ranks_file, 'fulllex')

    # asses_composition(path_observed_embeddings, addmask_predictions_file, addmask_ranks_file, 'addmask')
    # asses_composition(path_observed_embeddings, wmask_predictions_file, wmask_ranks_file, 'wmask')

