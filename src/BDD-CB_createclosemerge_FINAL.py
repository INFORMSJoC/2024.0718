import networkx as nx
import numpy as np
import pickle
#import matplotlib.pyplot as plt
from graphviz import Digraph
import os
import random
import shutil
import csv
import time

import pandas as pd 


''' 

Final version for IJOC Publication.

Description of Code: This code reads the capital budgeting instances and creates either an exact or approximate, unordered BDD.
It then saves (i) the time it took to generate the BDD, (ii) the number of arcs in the unreduced BDD, and (iii) the BDD itself.


How to use:
   - for approximate DDs, merge_param defines the absolute difference in the value of states that we are willing to accept when merging. 
   -  approx_DD = False will generate an exact decision diagram.
   -  approx_DD = True will generate an approximate decision diagram.
       - when we use this setting, we need to decide whether we want an inner or outer approximation.
       - we use tag = '_relQ' to define a relaxation (outer approx) with value of Q given by merge_param. 
           - when we are merging, we need to ensure merged_state = min(.)
       - we use tag = '_resQ' to define a restriction (inner approx) with value of Q given by merge_param.
           - when we are merging, we need to ensure merged_state = max(.)

'''

# -- file of different instances

master_folder = 'N30 Instances budget200'
to_folder = 'N30 Instances budget200 BDDs'

reverse_order = False	# reverse variable ordering (ignore)
approx_DD = False
merge_param = 10    # number of units of difference in states that we are willing to accept when merging

if approx_DD == True:
	tag = '_relQ' + str(merge_param)
else:
	tag = ''



def get_key_by_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def node_merge_technique(lns_dic, merge_param):
	# input: lns_dic: a dictionary of nodes at this layer (key), and the states of each node (value)
	# input: nodes_nextlayer = list of node references
	# input: num_to_merge = scalar value that will always be less than the number of nodes_nextlayer
	# output: a subset of nodes_nextlayer, where the first entry will be the one that all nodes are merged to

	nodes = list(lns_dic.keys())    # list of node references 
	states = list(lns_dic.values())
	sorted_states = sorted(states)
	nodesets_to_merge = []
	current_min_state = sorted_states[0]
	nodeset = []


	for state in sorted_states:
		if state - current_min_state <= merge_param:
			nodeset.append(get_key_by_value(lns_dic, state))
			if state == max(sorted_states):
				nodesets_to_merge.append(nodeset)
		else:
			nodesets_to_merge.append(nodeset)
			current_min_state = state
			nodeset = []
			nodeset.append(get_key_by_value(lns_dic, state))
			if state == max(sorted_states):
				nodesets_to_merge.append([get_key_by_value(lns_dic, state)])



	return nodesets_to_merge


for file_x in os.listdir(master_folder): 	# for a given instance, do the following 

	start_time = time.time()

	if file_x.startswith('.'):
		continue


	instance_name = os.fsdecode(file_x) 
	file_name = master_folder + '/' + instance_name
	file=open(file_name)



	# - assinging or making a directory

	file_path = to_folder + '/' + 'BDD_' + instance_name
	if not os.path.exists(file_path):
		os.makedirs(file_path)


	# read first line of the instance

	first_line = file.readline().split(' ')
	first_line = [i.strip('\n') for i in first_line]
	first_line = [float(i) for i in first_line]            # convert strings to numerics
	print(first_line)

	n_items = first_line[0]
	capacity = first_line[1]
	C_1 = first_line[2]
	C_2 = first_line[3]
	lambd = first_line[4]
	lambmu = first_line[5]
	f = first_line[6]
	M = first_line[8] #number of risk factors

	pbar_vec = []
	cbar_vec = []
	Q_matrix = []

	# read second until last line of the instance

	for line in file:
		current_line = line.split(' ')
		current_line = [i.strip('\n') for i in current_line]
		current_line.pop(0)                  # remove initial spacing in the instance file that starts on second line
		current_line = [float(i) for i in current_line]
		pbar_vec.append(current_line[0])     # objective values of each decision
		cbar_vec.append(current_line[1])	 # weight of each decision
		Q_matrix.append(current_line[2:])
		print(current_line)

	# the following code used only if we want to randomly do variable ordering: 
	# if reverse_order == False, then this block returns weight_vector = cbar_vec.

	indicies = list(range(int(n_items)))
	random_indicies = indicies
	weight_vector = []

	if reverse_order == True:
		random.shuffle(random_indicies)   #cbar_vec.reverse()

	for i in random_indicies:
		weight_vector.append(cbar_vec[i])

	with open(file_path + '/var_ordering%s.npy' % tag, 'wb') as f:
		np.save(f, random_indicies)



	# adds (-C_1, -C_2) to start of weight_vector
	weight_vector.insert(0,-C_2)   
	weight_vector.insert(0,-C_1)

	print('variable ordering')
	print(random_indicies)
	print('weight vector')
	print(weight_vector)  # [z_0, y_0, y_1, ... y_n]

	n_items = int(n_items + 2)   # because there are two variables related to first-stage and second-stage loans


	# ======= Building the BDD ========

	# state_update(): updates the current volume in the knapsack 
	# --- state_new = state_current + d * weight
	# --- d = {0 , 1}

	def state_update(state_current, d, weight, capacity):
		state_new = state_current + d*weight
		if state_new > capacity:
			return 'infeasible'
		else:
			return state_new




	# - Initialize the networkx graph
	# --- MultiDiGraph() means that there can be multiple, directed links between nodes

	G = nx.MultiDiGraph()




	# - Initialize dictionary tracking Layer: Node number: State (lns)
	# --- Example: lns_dic = {0: {'n1' : 0},
	#						  1: {'n1' : c_1, 'n2' : 0}}
	# --- layer in [0, ..., n_items]
	# --- layer i corresponds to knapsack after item i decision

	lns_dic = {}
	for layer in range(n_items + 1):
		lns_dic[layer] = {}

	# --- states for first and last node

	lns_dic[0] = {'n1': 0}
	lns_dic[n_items] = {'n1': capacity}






	# - Building the BDD, first n - 1 layers

	for layer in range(n_items-1):                   #layer: 1, 2, ... n
		print('layer ' + str(layer))								
		for node in lns_dic[layer]:                 #node: reference of node in the layer, e.g., "n1", "n2". Starts with root node "n1" at layer 0
			state_current = lns_dic[layer][node]    #state_current: state of the current node we are in

			for d in [0,1]:
				next_states_list = lns_dic[layer + 1].values()
				state_new = state_update(state_current, d, weight_vector[layer], capacity)     #state of the new node after adding d = {0,1}

				if state_new == 'infeasible':  # if we can't add the item (weight exceeds capacity)
					continue

				elif (state_new in next_states_list):   # if we add the item, and the new weight is already defined by a state in the next layer
					next_node = list(lns_dic[layer + 1].keys())[list(next_states_list).index(state_new)]   # next_node: the node in the next layer that already has the state
					G.add_edge(str(layer) + node, str(layer + 1) + next_node, d)       # add edge from node to next node

				else:    # otherwise, we have to create a new node 
					num_of_next_nodes = len(lns_dic[layer + 1].keys())      # number of existing nodes in next layer
					lns_dic[layer + 1]['n' + str(num_of_next_nodes+1)] = state_new      # add a new node with a new state to the dictionary
					G.add_edge(str(layer) + node, str(layer + 1) + 'n' + str(num_of_next_nodes+1), d)		# add an edge between current node to this new node

		print('number of nodes in layer is ' + str(len(list(lns_dic[layer + 1].keys()))))
		# - Merge nodes for approximate decision diagram
		if approx_DD == True:

			#print(' -- value of states --')
			#print(lns_dic[layer + 1].values())

			# added to prevent random merging, and instead to merging of lowest states
			nodesets_to_merge = node_merge_technique(lns_dic[layer + 1], merge_param)
			#print(nodesets_to_merge)

			for nodes_to_merge in nodesets_to_merge: 
				if len(nodes_to_merge) == 1:
					continue
				merged_node = nodes_to_merge[0]
				merged_state = min(lns_dic[layer+1][n] for n in nodes_to_merge)     # max gives restricted DD, min gives relaxed DD
				nodes_to_merge.remove(merged_node)
				#print(list(lns_dic[layer+1][n] for n in nodes_to_merge))
				#print(merged_state)
				lns_dic[layer+1][merged_node] = merged_state     # update merged_node to have the new max/min state 
				for node in nodes_to_merge:
					#print(node)
					in_edges = G.in_edges(str(layer+1) + node, keys = True)
					#print(in_edges)
					new_in_edges = [(source, str(layer+1) + merged_node, label) for source, sink, label in in_edges] 
					#print(new_in_edges)
					G.remove_node(str(layer + 1) + node)
					G.add_edges_from(new_in_edges)
					lns_dic[layer+1].pop(node, None)


	# - Connecting layer n - 1 to layer n

	layer = n_items - 1
	for node in lns_dic[layer]:
		state_current = lns_dic[layer][node]
		for d in [0,1]:
			state_new = state_update(state_current, d, weight_vector[layer], capacity)
			if state_new != 'infeasible':
				G.add_edge(str(layer) + node, str(layer + 1) + 'n1', d)



	end_time = time.time()
	total_time = end_time - start_time

	# - Save Time

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