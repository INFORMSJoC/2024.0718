import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from graphviz import Digraph
import os
import random
import shutil
import csv
import time

import pandas as pd 


''' 

Final version for IJOC Publication. 

Description of code: this code generates several exact decision diagrams for the assignment problem instances. 


'''

# -- file of different instances



master_folder = 'Assignment Instances 25-8'
to_folder = 'Assignment Multi BDDs 25-8'

for file_x in os.listdir(master_folder): 	 

	

	if file_x.startswith('.'):
		continue
	instance_name = os.fsdecode(file_x) 
	file_name = master_folder + '/' + instance_name


	# - assinging or making a directory

	file_path = to_folder + '/' + 'BDD_' + instance_name
	if not os.path.exists(file_path):
		os.makedirs(file_path)



	# Read instance

	y_agent_indices = []
	y_task_indices = []
	y_nominal_weights = []
	n_decisions = 0
	n_agents = 0
	n_tasks = 0


	with open(file_name, "r") as file:
		reader = csv.reader(file)
		row = next(reader)
		first_row = [float(i) for i in row]
		n_decisions, n_agents, n_tasks = int(first_row[0]), int(first_row[1]), int(first_row[2])

		row = next(reader)
		agent_weights = [int(i) for i in row]

		row = next(reader)
		capacity_vec = [int(i) for i in row]


		for row in reader:
			y_row = [float(i) for i in row]    # of the form [agent_index, task_index, nominal_reward]
			y_agent_indices.append(int(y_row[0]))
			y_task_indices.append(int(y_row[1]))
			y_nominal_weights.append(y_row[2])



	indices = list(range(int(n_decisions)))


	with open(file_path + '/var_ordering.npy', 'wb') as f:
		np.save(f, indices)




	# ======= Building the BDD ========

	# state_update(): updates the current volume in the knapsack 
	# --- state_new = state_current + state_addition
	# --- d = {0 , 1}

	for task in range(n_tasks):

		start_time = time.time()

		def state_update(state_current, layer, d, n_agents, n_tasks, agent_index, agent_weight, task_index):
			state_addition = 0   # tasks capacities, then agents assignments
			if task == task_index:
				state_addition = d*agent_weight

			state_new = state_current + state_addition
			#print(state_new)
			if state_new > capacity_vec[task]:
				return 'infeasible'
			else:
				if agent_index <= n_agents - 2:
					min_agent_weight = min(agent_weights[(agent_index+1):])
				else:
					min_agent_weight = capacity_vec[task] + 1
					if state_new + d*min_agent_weight + (1-d)*min(agent_weights[agent_index:]) > capacity_vec[task]:
						state_new = capacity_vec[task]
				return state_new



		# - Initialize the networkx graph
		# --- MultiDiGraph() means that there can be multiple, directed links between nodes

		G = nx.MultiDiGraph()



		# - Initialize dictionary tracking Layer: Node number: State (lns)
		# --- Example: lns_dic = {0: {'n1' : 0},
		#						  1: {'n1' : c_1, 'n2' : 0}}
		# --- layer in [0, ..., n_decisions]
		# --- layer i corresponds to knapsack after item i decision

		lns_dic = {}
		for layer in range(n_decisions + 1):
			lns_dic[layer] = {}

		# --- states for first and last node

		lns_dic[0] = {'n1': 0}
		lns_dic[n_decisions] = {'n1': capacity_vec[task]}



		# - Building the BDD, first n - 1 layers

		for layer in range(n_decisions-1):								 #layer: 1, 2, ... n
			print(layer)

			for node in lns_dic[layer]:                 #node: reference of node in the layer, e.g., "n1", "n2". Starts with root node "n1" at layer 0
				state_current = lns_dic[layer][node]    #state_current: state of the current node we are in

				for d in [0,1]:
					next_states_list = lns_dic[layer + 1].values()
					#print(next_states_list)
					state_new = state_update(state_current, layer, d, n_agents, n_tasks, y_agent_indices[layer],agent_weights[y_agent_indices[layer]], y_task_indices[layer])     #state of the new node after adding d = {0,1}

					if state_new == 'infeasible':  # if we can't add the item (weight exceeds capacity)
						continue

					elif (state_new in next_states_list):   # if we add the item, and the new weight is already defined by a state in the next layer
						next_node = list(lns_dic[layer + 1].keys())[list(next_states_list).index(state_new)]   # next_node: the node in the next layer that already has the state
						G.add_edge(str(layer) + node, str(layer + 1) + next_node, d)       # add edge from node to next node
						#print('yes')

					else:    # otherwise, we have to create a new node 
						num_of_next_nodes = len(lns_dic[layer + 1].keys())      # number of existing nodes in next layer
						lns_dic[layer + 1]['n' + str(num_of_next_nodes+1)] = state_new      # add a new node with a new state to the dictionary
						G.add_edge(str(layer) + node, str(layer + 1) + 'n' + str(num_of_next_nodes+1), d)		# add an edge between current node to this new node




		# - Connecting layer n - 1 to layer n

		layer = n_decisions - 1
		for node in lns_dic[layer]:
			state_current = lns_dic[layer][node]
			for d in [0,1]:
				state_new = state_update(state_current, layer, d, n_agents, n_tasks, y_agent_indices[layer],agent_weights[y_agent_indices[layer]], y_task_indices[layer])
				if state_new != 'infeasible':
					G.add_edge(str(layer) + node, str(layer + 1) + 'n1', d)



		end_time = time.time()
		total_time = end_time - start_time

		# - Save Time

		tag = str(task)

		f = open(file_path + '/DD_gen_time%s.txt' % tag, 'w')
		f.write(str(total_time))
		f.close()

		# - Save Graph and Layer-node-state dictionary


		# if reverse_order == True:
		nx.write_gpickle(G, file_path + '/DD_unreduced%s.pkl' % tag)
		with open(file_path + '/lns_dic%s.pkl' % tag, 'wb') as f:
			pickle.dump(lns_dic, f)

		# - Save Graph states

		# lines = ["number of arcs: " + str(len(list(G.edges)))]
		f = open(file_path + '/num_arcs_unreduced%s.txt' % tag, 'w')
		f.write(str(len(list(G.edges))))
		f.close()
		# for line in lines:
		# 	f.write(line)
		# 	f.write("\n")



	print(os.getcwd())
	shutil.copy(master_folder + '/' + instance_name, file_path + '/' + instance_name)