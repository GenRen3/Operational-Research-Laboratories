#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:21:00 2019

@author: Francesca
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy

def SCOM(N,DELTA):

    np.random.seed(42)

    #input parameters
    nodes = np.arange(N)
    tsd = np.random.uniform(0, 4, (N,N))
    np.fill_diagonal(tsd,0)
    Tsd = copy.copy(tsd)

    #variables
    bij = np.zeros((N,N))
    degtx = np.zeros((N,1))
    degrx = np.zeros((N,1))
    fij = np.zeros((N,N))

    #SCOM greedy algorithm to find bij
    while ((np.max(Tsd)>0)):
        (s,d) = np.unravel_index(np.argmax(Tsd, axis=None), Tsd.shape)
        Tsd[s,d] = 0
        if (degtx[s]<DELTA and degrx[d]<DELTA):
            bij[s,d] = 1
            degtx[s] += 1
            degrx[d] += 1

    #graph creation
    elist = np.argwhere(bij)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(elist)
#    nx.draw(G)

    #route the traffic to compute fij
    for s in range (0,N):
        for d in range (0,N):

            try:
                my_path = nx.shortest_path(G,source = s, target = d)
            except:
                my_path = []

            for i in range (0,len(my_path)-1):
                fij[my_path[i],my_path[i+1]] = fij[my_path[i],my_path[i+1]]+tsd[s,d]

    print ('Nodes: ', N, 'Edges: ', G.number_of_edges(),'Fmax: ', np.max(fij))
    return (N,G.number_of_edges(),tsd,np.max(fij))

def generate_random_graph(N,DELTA):

    (nodes,edges,tsd,fmax) = SCOM(N,DELTA)
    results = []

    #quante volte bisognerà farlo per avere un valor medio accettabile?
    count = 0
    while count < 100:

        #per come ho scritto il codice è possibile che certe volte non raggiunga
        #in modo random una soluzione fattibile, quindi nel caso riprova
        ok=0

        while (ok == 0):
            degtx = np.zeros((nodes,1))
            degrx = np.zeros((nodes,1))
            bij = np.zeros((nodes,nodes))
            chosableij = np.ones((nodes,nodes))
            np.fill_diagonal(chosableij,0)
            fij = np.zeros((nodes,nodes))
            number_of_edges = 0

            try:
                #considera tutti gli edges come possibili e ne sceglie a caso
                #tanti quanti ne ha il grafo dello scom
                while (number_of_edges < edges):
                    np.random.seed() # dovrebbe servire ad averlo diverso ogni volta
                    possible_edges = np.argwhere(chosableij)
                    choice = np.random.randint(0,len(possible_edges))
                    s = possible_edges[choice,0]
                    d = possible_edges[choice,1]
                    chosableij[s,d] = 0
                    if (degtx[s]<DELTA and degrx[d]<DELTA):
                        bij[s,d] = 1
                        degtx[s] += 1
                        degrx[d] += 1
                        number_of_edges += 1
                ok = 1

            except:
                continue


        elist = np.argwhere(bij)
        G = nx.DiGraph()
        my_nodes = np.arange(0,nodes)
        G.add_nodes_from(my_nodes)
        G.add_edges_from(elist)
#        nx.draw(G)

        # faccio il routing come nello scom
        for s in range (0,N):
            for d in range (0,N):

                    try:
                        my_path = nx.shortest_path(G,source = s, target = d)
                    except:
                        my_path = []

                    for i in range (0,len(my_path)-1):
                        fij[my_path[i],my_path[i+1]] = fij[my_path[i],my_path[i+1]]+tsd[s,d]
        count += 1
        results.append(np.max(fij))

    return (np.mean(results),fmax)

if __name__ == '__main__':

    N = [20,30,40]
    DELTA = [1,2,4]

    results_scom = np.zeros((len(DELTA),len(N)))
    results_random = np.zeros((len(DELTA),len(N)))

    count_delta = 0
    count_N = 0

    for delta in DELTA:
        count_N = 0
        for n in N:
            (results_random[count_delta,count_N],results_scom[count_delta,count_N]) = (generate_random_graph(n,delta))
            count_N += 1
        count_delta += 1

#%%
    plt.figure()

    plt.plot(N,results_scom[0,:], 'ro-')
    plt.plot(N,results_scom[1,:], 'go-')
    plt.plot(N,results_scom[2,:], 'bo-')

    plt.plot(N,results_random[0,:], 'ro--')
    plt.plot(N,results_random[1,:], 'go--')
    plt.plot(N,results_random[2,:], 'bo--')

    plt.xlabel('nodes')
    plt.legend([r'SCOM $\Delta$=1', 'SCOM $\Delta$=2', 'SCOM $\Delta$=4',
                'Random $\Delta$=1','Random $\Delta$=2','Random $\Delta$=4',])
    plt.ylabel(r'fmax')
    title = 'fmax vs nodes'
    plt.xticks([20, 30, 40])
    plt.title(title)
#    plt.margins(0.01, 0.1)
    plt.grid(which='both', axis='y')
#    plt.savefig(str(title) + '.png')
    plt.show()
#    print (average_improvement,np.mean(average_improvement))
