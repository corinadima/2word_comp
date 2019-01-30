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

def compute_neighbours(chosen, path_composed_emb, path_observed_emb, no_neighbours):
    """
        Returns the neighbours of words from a composed space in an observed space.
    """
    nearest_neighbours = {}

    observed_space = Word2VecKeyedVectors.load_word2vec_format(path_observed_emb, binary=False)
    observed_space.vectors = normalize(observed_space.vectors, norm="l2", axis=1)

    composed_space = Word2VecKeyedVectors.load_word2vec_format(path_composed_emb, binary=False)
    composed_space.vectors = normalize(composed_space.vectors, norm="l2", axis=1)

    chosen_words = set([tup[0] for tup in chosen])

    composed_words = composed_space.wv.vocab
    observed_words = observed_space.wv.vocab

    for word, rank in chosen:
        original_vec = observed_space.get_vector(word)
        composed_vec = composed_space.get_vector(word)

        original_composed_cosine = np.dot(original_vec, composed_vec)
        neighbours = observed_space.similar_by_vector(vector=original_vec, topn=no_neighbours)
        neighbours.append(("%s\_c" % word, original_composed_cosine))
        sorted_neighbours = sorted(neighbours, key=lambda tup: tup[1], reverse=True)
        c_idx = [idx for idx, tup in enumerate(sorted_neighbours) if tup[0]=="%s\_c" % word]
        print(word, original_composed_cosine, c_idx)

        nearest_neighbours[word] = sorted_neighbours

    return nearest_neighbours

def compute_both_neighbours(chosen, path_composed_emb, path_observed_emb):
    """
        Returns the neighbours of the composed/observed representations of the chosen words 
        in an observed space.
    """
    original_nearest_neighbours = {}
    composed_nearest_neighbours = {}

    observed_space = Word2VecKeyedVectors.load_word2vec_format(path_observed_emb, binary=False)
    observed_space.vectors = normalize(observed_space.vectors, norm="l2", axis=1)

    composed_space = Word2VecKeyedVectors.load_word2vec_format(path_composed_emb, binary=False)
    composed_space.vectors = normalize(composed_space.vectors, norm="l2", axis=1)

    chosen_words = set([tup[0] for tup in chosen])

    composed_words = composed_space.wv.vocab
    observed_words = observed_space.wv.vocab

    for word, rank in chosen:
        original_vec = observed_space.get_vector(word)
        composed_vec = composed_space.get_vector(word)

        original_composed_cosine = np.dot(original_vec, composed_vec)
        sims = observed_space.similar_by_vector(vector=original_vec, topn=False)
        neighbours = [(observed_space.index2word[widx], sims[widx]) for widx in range(len(sims))]
        neighbours.append(("%s\_c" % word, original_composed_cosine))
        sorted_neighbours = sorted(neighbours, key=lambda tup: tup[1], reverse=True)
        print("neighbours of the original representation of %s" % (word))
        c_idx = [idx for idx, tup in enumerate(sorted_neighbours) if tup[0]=="%s\_c" % word]
        print(word, original_composed_cosine, c_idx)

        original_nearest_neighbours[word] = sorted_neighbours[:11]
        print(original_nearest_neighbours[word])
        comp_index = c_idx[0]
        if comp_index >= 5:
            composed_nearest_neighbours["%s\_c" % word] = sorted_neighbours[comp_index-5:comp_index+6]
        else:
            composed_nearest_neighbours["%s\_c" % word] = sorted_neighbours[:11-comp_index]
        print("neighbours of the composed representation of %s" % (word))
        print(composed_nearest_neighbours["%s\_c" % word])

    return original_nearest_neighbours, composed_nearest_neighbours


def compute_best_rank_average(best_rank, test_ranks, path_composed_emb, path_observed_emb):
    chosen_words = set([tup for tup in test_ranks if tup[1] == best_rank])
    print("chosen words", len(chosen_words))

    nearest_neighbours = compute_neighbours(chosen_words, path_composed_emb, path_observed_emb, 1)
    rank_cosines = [1]
    for word, lst in nearest_neighbours.items():
        rank_cosines.append(lst[1][1])

    avg_cosine = np.mean(rank_cosines)
    print(str(len(chosen_words)) + " words of rank " + str(best_rank))
    print("average cosine %.5f" % avg_cosine)


def latex_print_info(output_file, sample_size, chosen_examples, nearest_neighbours, neighbours):
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

        for i in range(math.floor(len(chosen_examples)/sample_size)):
            format_string = "\\textbf{%s:%d} & " * (sample_size - 1) + "\\textbf{%s:%d} \\\\\n "
            tp = chosen_examples[i*sample_size:(i+1)*sample_size]
            ctp = list(itertools.chain(*tp))
            out.write(format_string % tuple(ctp[:]))
            out.write("\n\midrule\n")

            current_words = [tup[0] for tup in tp]
            print("current_words", current_words)

            for j in range(len(nearest_neighbours[current_words[i]])):     
                format_string = "%s %.5f & " * (len(tp)-1) + "%s %.5f \\\\\n "
                nw = [nearest_neighbours[word][j][0] for word in current_words]
                nc = [nearest_neighbours[word][j][1] for word in current_words]
                ncw = zip(nw, nc)
                max_rank = max([tup[1] for tup in tp])
                if max_rank > neighbours-1 and j == neighbours:
                    format_dots = "%s & " * (len(tp)-1) + "%s \\\\\n"
                    dots = ["..." for _ in tp]
                    out.write(format_dots % tuple(dots))

                out.write(format_string % tuple(list(itertools.chain(*ncw))))
            out.write("\n\midrule")

        out.write("\end{tabular}\n")
        out.write("\caption{\label{ch5:table:de_nn-only_test_head_examples}}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")
        out.write("\end{document}\n")


def latex_print_info_both(output_file, chosen_examples, original_neighbours, composed_neighbours):
    sample_size = 4
    with open(output_file, mode='w', encoding='utf8') as out:
        out.write("\n")

        # out.write("\\documentclass{article}\n") 
        # out.write("\\usepackage[utf8]{inputenc}\n") 
        # out.write("\\usepackage{booktabs}\n")   
        # out.write("\\begin{document}\n")
        out.write("\\begin{table}[!tbh]\n")
        out.write("\\begin{center}\n")
        out.write("\\scriptsize\n")
        out.write("\\begin{tabular}{rrrr}\n")

        for i in range(math.floor(len(chosen_examples)/sample_size)):
            format_string = "\\textbf{%s:%d} & " * (sample_size - 1) + "\\textbf{%s:%d} \\\\\n "
            tp = chosen_examples[i*sample_size:(i+1)*sample_size]
            ctp = list(itertools.chain(*tp))
            out.write(format_string % tuple(ctp[:]))
            out.write("\n\midrule\n")

            current_words = [tup[0] for tup in tp]
            format_string = "%s %.5f & " * (len(tp)-1) + "%s %.5f \\\\\n "
            format_dots = "%s & " * (len(tp)-1) + "%s \\\\\n"
            dots = ["..." for _ in tp]

            for j in range(len(original_neighbours[current_words[i]])):     
                nw = [original_neighbours[word][j][0] for word in current_words]
                nc = [original_neighbours[word][j][1] for word in current_words]
                ncw = zip(nw, nc)
                out.write(format_string % tuple(list(itertools.chain(*ncw))))

            out.write(format_dots % tuple(dots))

            for j in range(len(composed_neighbours["%s\_c" %current_words[i]])):     
                nw = [composed_neighbours["%s\_c" % word][j][0] for word in current_words]
                nc = [composed_neighbours["%s\_c" % word][j][1] for word in current_words]
                ncw = zip(nw, nc)
                out.write(format_string % tuple(list(itertools.chain(*ncw))))

            out.write("\n\midrule")

        out.write("\end{tabular}\n")
        out.write("\caption{\label{ch7:table:de_lex_examples}}\n")
        out.write("\end{center}\n")
        out.write("\end{table}\n")
        # out.write("\end{document}\n")


def select_sample(sample_size, start_rank, ranks_dict):
    samples = []
    sorted_keys = sorted(ranks_dict)
    current_list = random.sample(ranks_dict[start_rank], len(ranks_dict[start_rank]))
    current_key_idx = sorted_keys.index(start_rank)
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
    print(high_rank_list, best_rank)

    middle_rank = sorted_test[math.floor(len(sorted_test)/2)-int(math.floor(sample_size/2))][1]
    middle_rank_list = select_sample(sample_size, middle_rank, test_ranks_dict)
    print(middle_rank_list, middle_rank)

    lowest_rank = sorted_test[-1][1]
    low_rank_list = select_sample(sample_size, lowest_rank, test_ranks_dict) 
    print(low_rank_list, lowest_rank)

    chosen_examples = list(itertools.chain(high_rank_list, middle_rank_list, low_rank_list))

    neighbours = 6
    nearest_neighbours = compute_neighbours(chosen_examples, path_composed_emb, path_observed_emb, neighbours)
    compute_best_rank_average(best_rank, sorted_test, path_composed_emb, path_observed_emb)
    output_file = str(Path('data/results/German/' + model_name + '_analysis.tex'))
    latex_print_info(output_file, sample_size, chosen_examples, nearest_neighbours, neighbours)

if __name__=="__main__":

    # path of the full embeddings (original)
    path_observed_embeddings = str(Path('data/embeddings/German/decow14ax/raw/decow14ax_all_min_100_vectors_300dim.txt'))

    # loading ranks and prediction files (containing composed embeddings) for trained models
    head_ranks_file = str(Path('data/results/German/model_HeadOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test_rankedCompounds.txt'))
    head_predictions_file = str(Path('data/results/German/model_HeadOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test.pred'))

    modifier_ranks_file = str(Path('data/results/German/model_ModifierOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test_rankedCompounds.txt'))
    modifier_predictions_file = str(Path('data/results/German/model_ModifierOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test.pred'))

    addition_ranks_file = str(Path('data/results/German/model_Addition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test_rankedCompounds.txt'))
    addition_predictions_file = str(Path('data/results/German/model_Addition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test.pred'))

    mul_ranks_file = str(Path('data/results/German/model_Multiplication_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test_rankedCompounds.txt'))
    mul_predictions_file = str(Path('data/results/German/model_Multiplication_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test.pred'))

    w_addition_ranks_file = str(Path('data/results/German/model_WeightedAddition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-43_test_rankedCompounds.txt'))
    w_addition_predictions_file = str(Path('data/results/German/model_WeightedAddition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-43_test.pred'))

    lexfunc_ranks_file = str(Path('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-01_16-41_test_rankedCompounds.txt'))
    lexfunc_predictions_file = str(Path('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-01_16-41_test.pred'))

    fulladd_ranks_file = str(Path('data/results/German/model_FullAdd_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-16_test_rankedCompounds.txt'))
    fulladd_predictions_file = str(Path('data/results/German/model_FullAdd_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-16_test.pred'))

    dil_ranks_file = str(Path('data/results/German/model_Dilation_tanh_adagrad_batch100_cosine_l2_col_lr_0-1_2018-04-22_08-43_test_rankedCompounds.txt'))
    dil_predictions_file = str(Path('data/results/German/model_Dilation_tanh_adagrad_batch100_cosine_l2_col_lr_0-1_2018-04-22_08-43_test.pred'))

    matrix_ranks_file = str(Path('data/results/German/model_Matrix_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-50_test_rankedCompounds.txt'))
    matrix_predictions_file = str(Path('data/results/German/model_Matrix_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-50_test.pred'))

    fulllex_ranks_file = str(Path('data/results/German/model_FullLex_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-04_18-17_test_rankedCompounds.txt'))
    fulllex_predictions_file = str(Path('data/results/German/model_FullLex_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-04_18-17_test.pred'))

    addmask_ranks_file = str(Path('data/results/German/model_AddMask_tanh_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-22_16-58-03_test_rankedCompounds.txt'))
    addmask_predictions_file = str(Path('data/results/German/model_AddMask_tanh_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-22_16-58-03_test.pred'))

    wmask_ranks_file = str(Path('data/results/German/model_WMask_tanh_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-22_14-13-21_test_rankedCompounds.txt'))
    wmask_predictions_file = str(Path('data/results/German/model_WMask_tanh_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-22_14-13-21_test.pred'))

    multimatrix_ranks_file = str(Path('data/results/German/model_MultiMatrix_ReLU_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-23_11-51-21_test_rankedCompounds.txt'))
    multimatrix_predictions_file = str(Path('data/results/German/model_MultiMatrix_ReLU_adagrad_batch100_cosine_l2_row_lr_0-1_2018-06-23_11-51-21_test.pred'))

    asses_composition(path_observed_embeddings, head_predictions_file, head_ranks_file, 'head_only')
    asses_composition(path_observed_embeddings, modifier_predictions_file, modifier_ranks_file, 'modifier_only')
    asses_composition(path_observed_embeddings, addition_predictions_file, addition_ranks_file, 'addition')
    asses_composition(path_observed_embeddings, mul_predictions_file, mul_ranks_file, 'mul')

    asses_composition(path_observed_embeddings, w_addition_predictions_file, w_addition_ranks_file, 'w_addition')
    asses_composition(path_observed_embeddings, lexfunc_predictions_file, lexfunc_ranks_file, 'lexfunc')
    asses_composition(path_observed_embeddings, fulladd_predictions_file, fulladd_ranks_file, 'fulladd')
    asses_composition(path_observed_embeddings, dil_predictions_file, dil_ranks_file, 'dil')
    asses_composition(path_observed_embeddings, matrix_predictions_file, matrix_ranks_file, 'matrix')
    asses_composition(path_observed_embeddings, fulllex_predictions_file, fulllex_ranks_file, 'fulllex')

    asses_composition(path_observed_embeddings, addmask_predictions_file, addmask_ranks_file, 'addmask')
    asses_composition(path_observed_embeddings, wmask_predictions_file, wmask_ranks_file, 'wmask')
    asses_composition(path_observed_embeddings, multimatrix_predictions_file, multimatrix_ranks_file, 'multimatrix')
    
    output_file = str(Path('data/results/German/lex_analysis.tex'))
    chosen = [('nachtschatten', 1000), ('besenreiser', 1000), ('wertschätzung', 1000), ('tierkreis', 1000)]
    original_nearest_neighbours, composed_nearest_neighbours = compute_both_neighbours(chosen, multimatrix_predictions_file, path_observed_embeddings)
    latex_print_info_both(output_file, chosen, original_nearest_neighbours, composed_nearest_neighbours)
