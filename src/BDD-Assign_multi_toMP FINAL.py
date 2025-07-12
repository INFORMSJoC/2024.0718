import numpy as np
import pandas as pd
import gurobipy as gp
import graphviz
from gurobipy import GRB
import networkx as nx
import pickle
from pdb import *
import os
import csv


'''

Final version for IJOC Publication.

Description of Code: This codes reads several decision diagrams, formulates the multi-network flow problem, solves it,
and saves (i) runtime, (ii) optimal objective value, and (iii) optimal first stage solution.

How to use: change master_folder with location of instance data and saved DDs.
'''



master_folder = 'Assignment Multi BDDs 25-8'
os.chdir('./' + master_folder)

for beta in [0.5, 0.6, 0.7, 0.8, 0.9]:

    #beta = 0.5
    tag = '' #'_LP'   #'_LP'
    beta_str = str(beta) + tag




    for file_x in os.listdir(os.getcwd()): 

        # go into instance folder
        if not file_x.startswith('BDD'):
            continue
        os.chdir('./' + file_x)
        print(file_x)


        # define the variable ordering
        with open('var_ordering.npy', 'rb') as f:
            variable_ordering = np.load(f)
        variable_ordering = [int(i) for i in variable_ordering]


        # load the parameters for the problem
        file_path = os.fsdecode(file_x)
        instance_file = str(file_path[4:])

        y_agent_indices = []
        y_task_indices = []
        y_nominal_weights = []

        with open(instance_file, "r") as file:
            reader = csv.reader(file)
            row = next(reader)
            first_row = [float(i) for i in row]
            n_decisions, n_agents, n_tasks = int(first_row[0]), int(first_row[1]), int(first_row[2])

            row = next(reader)
            agent_weights = [int(i) for i in row]

            row = next(reader)
            task_capacity = [int(i) for i in row]

            capacity_vec = np.append(task_capacity, np.ones(n_agents))

            for row in reader:
                y_row = [float(i) for i in row]    # of the form [agent_index, task_index, nominal_reward]
                y_agent_indices.append(int(y_row[0]))
                y_task_indices.append(int(y_row[1]))
                y_nominal_weights.append(-y_row[2]*agent_weights[int(y_row[0])])  #negative nominal weight * agent weight (since maximization problem)



        # - retrieve parameters describing graph
        # --- A_incidence is the incidence matrix (oriented means directed arcs)
        # --- b is the LHS of flow balance constraints [-1, 0, ..., 0, 1]
        # --- n_nodes, n_edges
        # --- n_vars is dimension of recourse problem, which is much smaller than n_nodes/n_edges

        n_vars = n_decisions

        # load the decision diagram as a network 

        G = {}
        A_incidence ={}
        n_nodes = {}
        n_edges = {}
        b = {}


        for task in range(n_tasks):

            BDD_file = "DD_reduced%s" % task
            G[task] = nx.read_gpickle(BDD_file + '.pkl')


            A_incidence[task] = nx.incidence_matrix(G[task], oriented = True).toarray()
            n_nodes[task], n_edges[task] = A_incidence[task].shape

            b[task] = np.zeros(n_nodes[task])
            b[task][0] = -1
            b[task][n_nodes[task] - 1] = 1



        # --------  Defining the Optimization Model -------------------

        model = gp.Model()
        #model.Params.LogToConsole = 0
        #model.setParam('MIPFocus', 1)
        if tag == '_1popt':
            model.setParam('MIPGap', 0.01)
        if tag == '_2popt':
            model.setParam('MIPGap', 0.02)
        if tag == '_5popt':
            model.setParam('MIPGap', 0.05)

        # log file location
        model.params.LogFile = 'logfile%s.log' % tag #% slack_perc

        # --- Variable Definitions

        # Define first-stage x variables
        if tag == '_LP':
            x = model.addVars(n_vars, vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
            #p = model.addVars(n_vars, vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
        else: 
            x = model.addVars(n_vars, vtype = GRB.BINARY)
            #p = model.addVars(n_vars, vtype = GRB.BINARY)


        # Define original second-stage y variables
        y = model.addVars(n_vars, lb = 0, ub = 1)   # y vector corresponds to [z_0, y_0, y_1, ... , y_n]

        # Define uncertainty set dual variables
        lam_1 = model.addVars(n_vars, lb = 0)
        lam_2 = model.addVars(n_vars, lb = 0)
        pi = model.addVars(n_vars, lb = 0)
        alpha = model.addVar(lb = 0)


        # Define z (network flow) variables
        z = {}
        for task in range(n_tasks):
            z[task] = model.addMVar(shape = n_edges[task], vtype=GRB.CONTINUOUS, lb = 0, ub = 1)



        # --- Constraint Definitions

        # Add constraints on first-stage variables x (without P)
        # ---- add each task capacity constraint
        cons_list = [None for i in range(n_tasks)]   #delete
        for k in range(n_tasks): 
            lfs = []  #left-hand side of task capacity constraint
            for i in range(n_decisions):
                if y_task_indices[i] == k:
                    agent_index = y_agent_indices[i]
                    lfs.append(agent_weights[agent_index]*x[i])
            #model.addConstr(sum(lfs) <= task_capacity[k])
            #cons_list[k] = model.addConstr(sum(lfs) <= task_capacity[k])

        for j in range(n_agents):
            lfs = []   #left-hand side of task capacity constraint
            for i in range(n_decisions):
                if y_agent_indices[i] == j:
                    lfs.append(x[i])
            #model.addConstr(sum(lfs) <= 1)



        model.addConstr(sum([x[i] for i in range(n_vars)]) <= int(beta*n_decisions+0.1))


        # Add assignment constraints on agents

        for agent in range(n_agents):
            model.addConstr(sum((y_agent_indices[i]== agent)*y[i] for i in range(n_vars)) <= 1)


        # Add relationship between first-stage variables y and second-stage variables x
        for i in range(n_vars):
            #model.addConstr(x[i] + p[i] <= 1)
            model.addConstr(y[i] <= x[i])# + p[i])

        # a heuristic method to reduce computational time
        # for i in range(pzero_indices):
        #     model.addConstr(p[i] <= 0.01)

        # Define z constraints
        for task in range(n_tasks):
            model.addConstr(A_incidence[task] @ z[task] == b[task])



        # add constraints so that elements of z correspond to those of y
        # First, define a dictionary (solid_dic) of z indicies, then equate those indicies with y
        for task in range(n_tasks):

            solid_dic = {}    # Example: {0: 1, 1: 3, 2: 5, ...}
            for count, arc in enumerate(list(G[task].edges)):
                source_layer = arc[0].partition('n')[0]
                if arc[2] == 1:
                    if len(list(solid_dic.keys())) == 0 or (source_layer not in list(solid_dic.keys())):
                        solid_dic[source_layer] = [count]
                    else:
                        solid_dic[source_layer].append(count)


            # without variable reordering 
            for index in range(n_vars):
                if str(index) in solid_dic:
                    model.addConstr(y[index] == gp.quicksum([z[task][i] for i in solid_dic[str(index)]]))
                else:
                    model.addConstr(y[index] == 0)



        # Add uncertainty set constraints as dual variable constraints
        for i in range(n_decisions):
            model.addConstr(lam_1[i] - lam_2[i] == y_nominal_weights[i]*y[i])
            model.addConstr(pi[i] - lam_1[i] - lam_2[i] + alpha == 0)


        # set objective
        model.setObjective(0.1*n_decisions*alpha + sum([lam_1[i]-lam_2[i]+0.5*pi[i] for i in range(n_vars)]), GRB.MINIMIZE)

        model.setParam("TimeLimit", 1802)

        model.optimize()

        runtime = model.Runtime
        objective_value = model.objVal
        x_values = list([x[i].X for i in range(len(x))])
        #p_values = list([p[i].X for i in range(len(p))])
        #f_values = list([x_values[i] + p_values[i] for i in range(len(x_values))])
        f_values = list([x_values[i] for i in range(len(x_values))])



        f = open('new_runtime%s.txt' % beta_str, 'w')
        f.write(str(runtime))
        f.close()

        f = open('new_objval%s.txt' % beta_str, 'w')
        f.write(str(-objective_value))
        f.close()

        f = open('new_firststagesol%s.txt' % beta_str, 'w')
        for x in f_values:
            f.write(str(int(x+0.01)) + "\n")

        os.chdir('../')


