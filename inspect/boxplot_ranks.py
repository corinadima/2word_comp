#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.rcsetup as rcsetup
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plot_ranks(output_file):
    fulllex_10 = pd.read_csv('data/results/German/model_FullLex_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-04_18-17_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    matrix_9 = pd.read_csv('data/results/German/model_Matrix_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-50_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    dilation_8 = pd.read_csv('data/results/German/model_Dilation_tanh_adagrad_batch100_cosine_l2_col_lr_0-1_2018-04-22_08-43_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    fulladd_7 = pd.read_csv('data/results/German/model_FullAdd_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-16_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    lexfunc_6 = pd.read_csv('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-01_16-41_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    w_addition_5 = pd.read_csv('data/results/German/model_WeightedAddition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-43_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    mul_4 = pd.read_csv('data/results/German/model_Multiplication_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    addition_3 = pd.read_csv('data/results/German/model_Addition_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-42_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    modifier_2 = pd.read_csv('data/results/German/model_ModifierOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    head_1 = pd.read_csv('data/results/German/model_HeadOnly_tanh_adagrad_batch100_cosine_l2_col_lr_0-01_2018-04-22_08-41_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])


    indices = range(1, len(head_1)+1)

    fig = plt.figure()
    fig.set_size_inches(9, 3.5)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.tick_params(labelsize=6)
    ax.yaxis.set_ticks(np.arange(0,1001,200))

    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    spines_to_keep = ['bottom', 'left']
    for spine in spines_to_keep:
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('5410 nn-only test set compounds', fontsize=8)
    ax.set_ylabel('rank', fontsize=8)

    markE = 10

    data = np.transpose(np.vstack((mul_4['rank'], modifier_2['rank'], head_1['rank'], dilation_8['rank'], addition_3['rank'], w_addition_5['rank'], lexfunc_6['rank'], fulladd_7['rank'], matrix_9['rank'], fulllex_10['rank'])))
    print(data.shape)
    flierprops = dict(marker='o', markersize=2, linestyle='none', markeredgecolor='green', linewidth=0.1)
    ax.boxplot(data,  labels=['mul', 'modifier', 'head', 'dilation', 'addition', 'w_addition', 'lexfunc', 'fulladd', 'matrix', 'fulllex'], flierprops=flierprops)

    plt.ylim([-20, 1020])
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", dest="output_file", type=str, help="path for writing the ranks plot")

    args = parser.parse_args()

    plt.ioff()
    plot_ranks(args.output_file)

