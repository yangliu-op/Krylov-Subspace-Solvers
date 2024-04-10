# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:08:32 2024

@author: Yang
"""

import matplotlib.pyplot as plt
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

global colors, linestyles, markers
colors = [ 'k', 'b', 'm',  'c','g', 'r', 'y']
linestyles = ['--', '-',  '--', '-',   '-.', ':', ':', ':']
markers = ['o', '*', 'x', '1', '2', '+', 'o']
markers1 = ['o', 'X', '*', 'P', '1', '2', '.']
linewidth1 = [1.8, 1.8, 1.8, 1.8]
markersize1 = [3, 3, 4, 3]

def showFigure(methods_all, record_all, mypath):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, is_negative_curvature]
        prob: name of problem
        mypath: directory path for saving plots
    """
    fsize = 14
    myplt = plt.plot
    
    figsz = (6,3)
    mydpi = 300
    fig1 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i].clone()
        yaxis = record[:,0]
        xaxis = torch.tensor(range(0,len(yaxis)))
        draw(myplt, record, methods_all[i], i+1, (record.shape[1]==6), 
             yaxis, xaxis=xaxis)
    plt.ylabel(r'$ || x_t - x^{+} || / || x^{+} || $', fontsize=fsize)
    # plt.yscale('symlog')
    plt.grid(True)
    plt.legend()
    fig1.savefig(os.path.join(mypath, 'pseudo_xt'), dpi=mydpi)
            
    fig2 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        record = record_all[i].clone()
        yaxis = record[:,1]
        xaxis = torch.tensor(range(0,len(yaxis)))
        draw(myplt, record, methods_all[i], i+1, (record.shape[1]==6), yaxis, xaxis=xaxis)
    plt.ylabel(r'$ || \hat{x}_t^{\natural} - x^{+} || / || x^{+} ||$', fontsize=fsize)
    # plt.yscale('symlog')
    plt.grid(True)
    plt.legend()
    fig2.savefig(os.path.join(mypath, 'pseudo_hxt'), dpi=mydpi)
        
        
def draw(plt_fun, record, label, i, NC, yaxis, xaxis=None, index=None):
    if not (xaxis is not None):
        xaxis = torch.tensor(range(0,len(yaxis)))
    plt_fun(xaxis, yaxis, color=colors[i], linewidth=linewidth1[i], linestyle=linestyles[i], 
            marker=markers1[i], markersize=markersize1[i], label = label)
    if NC:
        if index is None:
            index = (record[:,5] == True)
        xNC = xaxis[index]
        yNC = yaxis[index]
        plt_fun(xNC, yNC, '.', color=colors[i], 
                marker=markers1[i], markersize=markersize1[i]*2.4)