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
    fulllex_10 = pd.read_csv('data/results/German/model_FullLex_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-22_08-46_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    matrix_9 = pd.read_csv('data/results/German/model_Matrix_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-50_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    dilation_8 = pd.read_csv('data/results/German/model_Dilation_tanh_adagrad_batch100_cosine_l2_col_lr_0-1_2018-04-22_08-43_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    fulladd_7 = pd.read_csv('data/results/German/model_FullAdd_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_10-16_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
    lexfunc_6 = pd.read_csv('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_15-44_test_rankedCompounds.txt', delimiter=' ', names=["compound", "rank"])
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

    ax.plot(indices, head_1['rank'],  label='head (baseline)', marker='s', linestyle=' ', markersize=1, markeredgecolor='#30333A', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, modifier_2['rank'],  label='modifier (baseline)', marker='x', linestyle=' ', markersize=1, markeredgecolor='#EC8305', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, mul_4['rank'],  label='mul', marker='d', linestyle=' ', markersize=1, markeredgecolor='#AC1D09', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, dilation_8['rank'],  label='dilation', marker='s', linestyle=' ', markersize=1, markeredgecolor='#34A64C', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, addition_3['rank'],  label='addition', marker='1', linestyle=' ', markersize=1, markeredgecolor='#BA036C', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, w_addition_5['rank'],  label='w_addition', marker='+', linestyle=' ', markersize=1, markeredgecolor='#13A2D4', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, lexfunc_6['rank'],  label='lexfunc', marker='H', linestyle=' ', markersize=1, markeredgecolor='#F4A066', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, fulladd_7['rank'],  label='fulladd', marker='o', linestyle=' ', markersize=1, markeredgecolor='#7F3BA7', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, matrix_9['rank'],  label='matrix', marker='+', linestyle=' ', markersize=1, markeredgecolor='#5B79FD', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    ax.plot(indices, fulllex_10['rank'],  label='fulllex', marker='x', linestyle=' ', markersize=1, markeredgecolor='#359A8E', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    # ax.plot(indices, addmask_11['rank'],  label='11. Addmask', marker='^', linestyle=' ', markersize=1, markeredgecolor='#9FBB77', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)
    # ax.plot(indices, wmask_12['rank'], label='12. Wmask (best)', marker='o', linestyle=' ', markersize=1, markeredgecolor='#F32831', markeredgewidth=0.25, markerfacecolor='None', markevery=markE, antialiased=True)


    # draw line at rank 5
    plt.axhline(y=5, linewidth=0.5, color='r', ls='dashed')
    ax.annotate('rank 5', xy=(5442, 4.5), xytext=(5455, 4.5),fontsize=6)

    # draw lines for quartiles
    plt.axvline(x=1361, linewidth=0.5, color='#B0DB76', ls='dotted')
    ax.annotate('25%', xy=(1361, 999), xytext=(1366, 999),fontsize=6)
    plt.axvline(x=2722, linewidth=0.5, color='#88AE3B', ls='dotted')
    ax.annotate('50%', xy=(2722, 999), xytext=(2727, 999),fontsize=6)
    plt.axvline(x=4083, linewidth=0.5, color='#6F9320', ls='dotted')
    ax.annotate('75%', xy=(4083, 999), xytext=(4088, 999),fontsize=6)

    arrow_width=0.5
    # annotate model names
    ax.annotate('mul',
            xy=(50, 700), xycoords='data',
            xytext=(20, -20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('modifier',
            xy=(2705, 400), xycoords='data',
            xytext=(-20, 20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('head',
            xy=(3400, 200), xycoords='data',
            xytext=(-20, 20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('dilation',
            xy=(3800, 200), xycoords='data',
            xytext=(-30, 40), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('addition',
            xy=(4300, 270), xycoords='data',
            xytext=(-45, 70), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('w_addition',
            xy=(4330, 300), xycoords='data',
            xytext=(0, -20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('lexfunc',
            xy=(4370, 150), xycoords='data',
            xytext=(0, -20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('fulladd',
            xy=(5100, 50), xycoords='data',
            xytext=(-30, 15), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('matrix',
            xy=(5200, 70), xycoords='data',
            xytext=(-20, 35), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))
    ax.annotate('fulllex',
            xy=(5220, 70), xycoords='data',
            xytext=(10, 20), textcoords='offset points', fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=arrow_width))

    legend = ax.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0., prop={'size':8}, numpoints=5)
    rect = legend.get_frame()
    rect.set_linewidth(0.5)

    plt.ylim([-20, 1020])
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", dest="output_file", type=str, help="path for writing the ranks plot")

    args = parser.parse_args()

    plt.ioff()
    plot_ranks(args.output_file)

