#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.rcsetup as rcsetup
import matplotlib.pyplot as plt
import pandas as pd

def plot_learning_curves(output_file):
    lexfunc_dr0 = pd.read_csv('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-04-23_15-44', delimiter=' ', names=["dev error", "training error"], header=0)
    lexfunc_dr05 = pd.read_csv('data/results/German/model_LexicalFunction_tanh_adagrad_batch100_cosine_l2_row_lr_0-01_2018-05-01_16-41', delimiter=' ', names=["dev error", "training error"], header=0)

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.tick_params(labelsize=6)
    # ax.yaxis.set_ticks(np.arange(0,max(len(lexfunc_dr0), len(lexfunc_dr05)),5))

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('epochs', fontsize=8)
    ax.set_ylabel('loss (cosine distance)', fontsize=8)

    dr0_index = range(len(lexfunc_dr0))
    dr05_index = range(len(lexfunc_dr05))

    linewidth = 1

    ax.plot(dr0_index, lexfunc_dr0['dev error'].astype(float), label='dr. 0: dev error', linewidth=linewidth)
    ax.plot(dr0_index, lexfunc_dr0['training error'].astype(float), label='dr. 0: train error', linewidth=linewidth, dashes=[6, 2])    
    # ax.plot(dr05_index, lexfunc_dr05['dev error'].astype(float), label='dr. 0.5: dev error', linewidth=linewidth, dashes=[2, 2, 10, 2])
    # ax.plot(dr05_index, lexfunc_dr05['training error'].astype(float), label='dr. 0.5: train error', linewidth=linewidth, dashes=[2, 2, 2, 2])

    legend = ax.legend(bbox_to_anchor=(0.95, 0.5), loc='center right', borderaxespad=0., prop={'size':8}, numpoints=5)
    rect = legend.get_frame()
    rect.set_linewidth(0.5)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", dest="output_file", type=str, help="path for writing the ranks plot")

    args = parser.parse_args()

    plt.ioff()
    plot_learning_curves(args.output_file)
