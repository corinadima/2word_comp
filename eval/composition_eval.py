#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import operator
from pprint import pprint as pp
import argparse

from sklearn.preprocessing import normalize

from gensim.models.keyedvectors import Vocab, Word2VecKeyedVectors

def computeQuartiles(l):
    """Return sample quartiles.
    Method 1 from https://en.wikipedia.org/wiki/Quartile:
    Use the median to divide the ordered data set into two halves. Do not include the median in either half.
    The lower quartile value is the median of the lower half of the data. The upper quartile value is the median of the upper half of the data.
    """

    med = np.median(l)
    sortedL = np.sort(l)
    n = len(l)
    midIndex = n // 2
    q1_list = sortedL[0: midIndex]
    if (n%2 == 0):
        q3_list = sortedL[midIndex:]
    else:
        q3_list = sortedL[midIndex+1:]
    q1 = np.median(q1_list)
    q3 = np.median(q3_list)
    return q1, med, q3

def targetBasedRank(targets, composed_model, original_model, max_rank=1000):
    """
    Computes the ranks of the composed representations, with respect to a dictionary of original embeddings. 
    The ordering is relative to the target representation.

    :param targets: the words to compute the ranks for
    :param composed_model: a gensim model containing the composed representations
    :param original_model: a gensim model containing the original representations
    :param max_rank: the maximum rank
    :return: a list with the ranks for all the composed representations in the batch 
    """
    rank_list = []
    word_ranks = {}

    # normalize both spaces, then the cosine similarity is the same as the dot product
    target_composed_idxs = [composed_model.vocab[w].index for w in targets]
    composed_repr = normalize(np.take(composed_model.vectors, target_composed_idxs, axis=0), 
            norm="l2", axis=1)
    original_repr = normalize(original_model.vectors, norm="l2", axis=1)

    target_original_idxs = [original_model.vocab[w].index for w in targets]
    target_repr = np.take(original_repr, target_original_idxs, axis=0)
    target_dict_similarities = np.dot(original_repr, 
                                np.transpose(target_repr))

    for i in range(len(targets)):
        # compute similarity between the target and the predicted vector
        target_composed_similarity = np.dot(composed_repr[i], target_repr[i])

        # remove the similarity of the target vector to itself
        target_sims = np.delete(target_dict_similarities[:, i], target_original_idxs[i])

        # the rank is the number of vectors with greater similarity that the one between
        # the target representation and the composed one; no sorting is required, just 
        # the number of elements that are more similar
        rank = np.count_nonzero(target_sims > target_composed_similarity) + 1
        if (rank > max_rank):
            rank = max_rank
        rank_list.append(rank)
        word_ranks[targets[i]] = rank
    return rank_list, word_ranks

def evaluateRank(targets, composed_space, original_space, max_rank):
    rankList, ranks = targetBasedRank(targets, composed_space, original_space, max_rank)
    q1, q2, q3 = computeQuartiles(rankList)
    return q1, q2, q3, ranks

def printDictToFile(dictionary, fileName):
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1))
    with open(fileName, mode='w', encoding='utf8') as outDict:
        for key, value in sorted_x:
            outDict.write("%s %d\n" % (key, value))

def printListToFile(save_list, fileName):
    with open(fileName, mode='w', encoding='utf8') as out:
        for value in save_list:
            out.write("%d\n" % value)
    
def logResult(q1,q2,q3, filename):
    with open(filename, mode='w', encoding='utf8') as out:
        out.write("Q1: %d, Q2: %d, Q3: %d\n" % (q1, q2, q3))

def read_targets(filename):
    targets = []
    with open(filename, mode='r', encoding='utf8') as inFile:
        next(inFile)
        for line in inFile:
            splits = line.strip().split(" ")
            targets.append(splits[0])
    return targets

def eval_on_file(path_composed_emb, path_observed_emb, save_path):
    raw_observed_space = Word2VecKeyedVectors.load_word2vec_format(path_observed_emb, binary=False)

    targets = read_targets(path_composed_emb)
    raw_composed_space = Word2VecKeyedVectors.load_word2vec_format(path_composed_emb, binary=False)

    q1, q2, q3, ranks = evaluateRank(targets, raw_composed_space, raw_observed_space, 1000)
    print("Q1: " + str(q1) + ", Q2: " + str(q2) + ", Q3: " + str(q3))

    if save_path:
        printDictToFile(ranks, save_path + '_rankedCompounds.txt')
        
        sortedRanks = sorted(ranks.values())
        printListToFile(sortedRanks, save_path + '_ranks.txt')
        logResult(q1, q2, q3, save_path + '_quartiles.txt')

    return q1, q2, q3, ranks
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-composed", dest="composed_file", type=str, 
        help="file containing the composed representations")
    parser.add_argument("-dictionary", dest="dict_file", type=str, 
        help="file containing the original representations (not only compounds, but also heads/modifiers)")
    parser.add_argument("-output", dest="ranks_file", type=str, help="path for writing the ranks file")
    parser.add_argument("--normalization", type=str, help="normalization type", default="")

    args = parser.parse_args()
    eval_on_file(path_composed_emb=args.composed_file, path_observed_emb=args.dict_file, 
        save_path=args.ranks_file)
