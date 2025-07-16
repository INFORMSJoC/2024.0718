import numpy as np
import pandas as pd
import gurobipy as gp
import graphviz
from gurobipy import GRB
import networkx as nx
import pickle
from pdb import *
import os


'''
Final version for IJOC Publication.

Description of Code: this code takes in a BDD and formulates and solves the two-stage capital budgeting problem with it.
It then saves (i) the runtime of the solver, (ii) the optimal objective value, and (iii) the first-stage solution.


'''

master_folder = 'N20 New BDDs'
tag = '_relQ10'  #res = restricted, rel = relaxed


os.chdir('./' + master_folder)


for file_x in os.listdir(os.getcwd()): 

    # go into instance folder
    if file_x.startswith('.') or file_x.startswith('D') or file_x.startswith('N'):
        continue

    os.chdir('./' + file_x)


    # load the decision diagram as a network 
    BDD_file = "DD_reduced%s" % tag
    G = nx.read_gpickle(BDD_file + '.pkl')


    # define the variable ordering
    with open('var_ordering.npy', 'rb') as f:
        variable_ordering = np.load(f)
    variable_ordering = [int(i) for i in variable_ordering]


    # load the parameters for the problem
    file_path = os.fsdecode(file_x)
    instance_file = str(file_path[4:])
    file=open(instance_file)

    first_line = file.readline().split(' ')
    first_line = [i.strip('\n') for i in first_line]
    first_line = [float(i) for i in first_line]

    print(first_line[0])
    print(int(first_line[0]))
    n_items = int(first_line[0])
    capacity = first_line[1]
    C_1 = first_line[2]
    C_2 = first_line[3]
    gamma = first_line[4]
    gammamu = first_line[5]
    f = first_line[6]
    M = int(first_line[8]) #number of risk factors

    pbar_vec = []
    cbar_vec = []
    Q_matrix = []


    for line in file:
        current_line = line.split(' ')
        current_line = [i.strip('\n') for i in current_line]
        current_line.pop(0)
        current_line = [float(i) for i in current_line]
        pbar_vec.append(current_line[0])        # objective values of each decision
        cbar_vec.append(current_line[1])        # weight of each decision
        Q_matrix.append(current_line[2:])

    n_items = int(n_items + 2)


    # - retrieve parameters describing graph
    # --- A_incidence is the incidence matrix (oriented means directed arcs)
    # --- b is the LHS of flow balance constraints [-1, 0, ..., 0, 1]
    # --- n_nodes, n_edges
    # --- n_vars is dimension of recourse problem, which is much smaller than n_nodes/n_edges
    A_incidence = nx.incidence_matrix(G, oriented = True).toarray()
    n_nodes, n_edges = A_incidence.shape

    n_vars = n_items
    b = np.zeros(n_nodes)
    b[0] = -1
    b[n_nodes - 1] = 1



    # --------  Defining the Optimization Model -------------------

    model = gp.Model()
    #model.Params.LogToConsole = 0
    #model.setParam('MIPFocus', 1)

    # log file location
    model.params.LogFile = 'logfile%s.log' % tag #% slack_perc

    # Set the MIP gap tolerance to 1e^-5 (default is 1e^-4)%
    model.Params.MIPGap = 0.00001

    # --- Variable Definitions

    # Define first-stage x variables
    if tag == '_LP':
        x = model.addVars(n_vars-1, vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
    else: 
        x = model.addVars(n_vars-1, vtype = GRB.BINARY)


    # Define original second-stage y variables
    y = model.addVars(n_vars, lb = 0, ub = 1)   # y vector corresponds to [z_0, y_0, y_1, ... , y_n]

    # Define uncertainty set dual variables
    lam_1 = model.addVars(M, lb = 0)
    lam_2 = model.addVars(M, lb = 0)


    # Define z (network flow) variables
    z = model.addMVar(shape = n_edges, vtype=GRB.CONTINUOUS, lb = 0, ub = 1)



    # --- Constraint Definitions

    # Add constraints on first-stage variables x
    model.addConstr(sum([cbar_vec[i]*x[i+1] for i in range(n_vars-2)]) - C_1*x[0] <= capacity)




    # Add relationship between first-stage variables y and second-stage variables x
    for i in range(n_vars-2):
        model.addConstr(y[i+2] >= x[i+1])

    model.addConstr(y[0] == x[0])

    # Define z constraints
    model.addConstr(A_incidence @ z == b)



    # add constraints so that elements of z correspond to those of x
    # First, define a dictionary (solid_dic) of z indicies, then equate those indicies with x
    solid_dic = {}    # Example: {0: 1, 1: 3, 2: 5, ...}
    for count, arc in enumerate(list(G.edges)):
        source_layer = arc[0].partition('n')[0]
        if arc[2] == 1:
            if len(list(solid_dic.keys())) == 0 or (source_layer not in list(solid_dic.keys())):
                solid_dic[source_layer] = [count]
            else:
                solid_dic[source_layer].append(count)


    # without variable reordering 
    # for index in range(n_vars):
    #     if str(index) in solid_dic:
    #         model.addConstr(y[index] == gp.quicksum([z[i] for i in solid_dic[str(index)]]))
    #     else:
    #         model.addConstr(y[index] == 0)
    #     print(index)

    # with variable reordering
    for index in range(2):
        if str(index) in solid_dic:
            model.addConstr(y[index] == gp.quicksum([z[i] for i in solid_dic[str(index)]]))
        else:
            model.addConstr(y[index] == 0)

    for index in range(2,n_vars):
        if str(index) in solid_dic:
            model.addConstr(y[variable_ordering[index-2]+2] == gp.quicksum([z[i] for i in solid_dic[str(index)]]))
        else:
            model.addConstr(y[variable_ordering[index-2]+2] == 0)



    # new: add original knapsack constraint on y
    # pureNF: comment this constraint out
    model.addConstr(gp.quicksum(cbar_vec[i]*y[i+2] for i in range(n_vars-2)) - C_1*y[0] - C_2*y[1] <= capacity)



    # Add uncertainty set constraints as dual variable constraints
    for m in range(M):
        model.addConstr(lam_1[m] - lam_2[m] == sum((1/2)*Q_matrix[i][m]*pbar_vec[i]*((1-f)*x[i+1]+f*y[i+2]) for i in range(n_vars-2)))


    # set objective
    model.setObjective(-gamma*x[0] + sum(pbar_vec[i]*((1-f)*x[i+1]+f*y[i+2]) for i in range(n_vars-2)) - (sum((lam_1[m] + lam_2[m]) for m in range(M))) - gammamu*y[1], GRB.MAXIMIZE)


    model.optimize()

    runtime = model.Runtime
    objective_value = model.objVal
    x_values = list([x[i].X for i in range(len(x))])

    f = open('runtime%s.txt' % tag, 'w')
    f.write(str(runtime))
    f.close()

    f = open('objval%s.txt' % tag, 'w')
    f.write(str(objective_value))
    f.close()

    f = open('firststagesol%s.txt' % tag, 'w')
    for x in x_values:
        f.write(str(int(x+0.01)) + "\n")

    os.chdir('../')


