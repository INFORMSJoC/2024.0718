import graphviz
import networkx as nx
import pickle
import os
import time
from collections import defaultdict




'''

Final version for IJOC publication.

Description of Code: Given a BDD, this algorithm reduces the BDD using a bottom-up approach. 
It then saves (i) the time it took to reduce the BDD, (ii) the number of arcs in the reduced BDD, and (iii) the BDD itself.

'''

# - Load the instance
# --- Load the graph G 
# --- Load the lns_dic to reference particular nodes in each layer

master_folder = 'N40 New BDDs'   # replace with folder name that contains all unreduced BDDs.
tag = '' #'_relQ10' 


os.chdir(os.getcwd() + '/' + master_folder)

for file_x in os.listdir(os.getcwd()):
	#print(file_x)

	start_time = time.time()

	if file_x.startswith('.') or file_x.startswith('D') or file_x.startswith('N'):
		continue

	file_path = os.fsdecode(file_x)


	G = nx.read_gpickle(file_path + '/DD_unreduced%s.pkl' % tag)
	lns_dic = nx.read_gpickle(file_path + '/lns_dic%s.pkl' % tag)

	# Precompute all outgoing and incoming edges

	#  lns_dic example (50 item knapsack): {0: {'n1': 0}, 1: {'n1': 0, 'n2': 11}, ... , 50: {'n1': 467}}

	# - Initialize, starting with last layer as sink_layer, and second last layer as source_layer

	maxarc_counter = 0


	for source_layer in reversed(range(0,len(lns_dic) - 1)):

		# source_layer example (50 item knapsack): source_layer goes from [49, 48, ..., 0]

		# nodes_source_layer is a list of node labels in the current layer
		# nodes_source_layer example (layer 49): ['49n1', '49n2', '49n3', ...]

		nodes_source_layer = [str(source_layer) + i for i in list(lns_dic[source_layer].keys())]

		out_edges_cache = {node: G.out_edges(node, keys = True) for node in nodes_source_layer}
		sink_and_label_cache = {node: [(sink, label) for source, sink, label in G.out_edges(node, keys=True)] for node in nodes_source_layer}
		in_edges_cache = {node: G.in_edges(node, keys = True) for node in nodes_source_layer}

		removed_nodes = set()
		new_edges = []
		nomerge_counter = 0  
		merge_counter = 0   
		totalarc_counter = 0

		#---------------

		value_to_keys = defaultdict(list)

		for count,node in enumerate(nodes_source_layer):
			out_edges = sink_and_label_cache[node]
			normalized_value = tuple(sorted(out_edges))  # Normalize the tuple
			value_to_keys[normalized_value].append(node)


		for value, keys in value_to_keys.items():
			if len(keys) >= 2:
				for key in keys[1:]:
					removed_nodes.add(key)
					in_edges_n2 = in_edges_cache[key]
					new_in_edges = [(source, keys[0], label) for source, sink, label in in_edges_n2]
					new_edges.extend(new_in_edges)

		G.remove_nodes_from(list(removed_nodes))
		G.add_edges_from(new_edges)      #add the new, redirected edges 

		 	

	print('max number of arcs in a node a layer' + str(maxarc_counter))
	end_time = time.time()
	total_time = end_time - start_time

	# - Save Time
	
	f = open(file_path + '/DD_reduce_timeV3%s.txt' % tag, 'w')
	f.write(str(total_time))
	f.close()

	# - Save graph states
	f = open(file_path + '/num_arcs_reducedV3%s.txt' % tag, 'w')
	f.write(str(len(list(G.edges))))
	f.close()


	# - Save reduced graph

	nx.write_gpickle(G, file_path + '/DD_reducedV3%s.pkl' % tag)





