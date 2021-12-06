import networkx as nx
from networkx.algorithms import bipartite

def refine (msq, id1, id2, verbose= False):
    q1 = msq[id1] # get msq (most specific query) for each id
    q2 = msq[id2] # get msq for each id

    objects1 = {i: c for i, c in enumerate (q1.concepts)} # objects in id1
    objects2 = {i + len(objects1): c for i, c in enumerate (q2.concepts)} # objects in id2

    # which concepts are common between the 2 instances
    matches = {}
    for i, c1 in objects1.items():
        for j, c2 in objects2.items():
            if c1 == c2 and j not in matches.values(): # if the objects are the same and this id is not already in the match, ie if the object has not already been matched with other object
                matches[i] = j
                break # stop searching because maybe the same object exists 2 times in the instance e.g. 2 persons or 2 times the same word

    # the match items must be removed from the list of items
    for i, j in matches.items():
        if verbose:
            print (f"Remains the same: {objects1[i]}")
        objects1.pop(i, None) # remove thsi object from the list of items of instance 1
        objects2.pop(j, None) # remove thsi object from the list of items of instance 2

    if verbose:
        print ("-------------------------------------")

    cost = 0 # we initialize the cost of transitions
    # now we are going to calculate the matches of the objects
    if len(objects1) != 0 and len(objects2) != 0: # if one of the 2 is empty we do not have to match so we just go and remove or add the objects of the other image
        B = nx.Graph() # create two bipartite graphs
        B.add_nodes_from(objects1.keys(), bipartite=0) # nodes are the objects in instance 1
        B.add_nodes_from(objects2.keys(), bipartite=1) # nodes are thw objects in instance 2
        # we want to add edges from obj in instance 1 with obj in instance 2
        # if a transition is not valid e.g. transform a old man to a young one
        # we just do not connect these 2 nodes

        transition_matrix = {}
        for i, obj1 in objects1.items(): # for each object in instance 1
            for j, obj2 in objects2.items(): # for each object in instance 2
                weight = obj_distance(obj1, obj2) # calculate the cost of the transition
                transition_matrix[f"{i}-{j}"] = weight
                B.add_edge(i, j, weight = weight) # add edge


        matches = bipartite.matching.minimum_weight_full_matching(B, objects1.keys(), "weight") # calculate minimum weight full matching
        for i, j in matches.items():
            if i in objects1: # to do it once because the matches return matches from both image 1 -> 2 and from 2 -> 1
                if transition_matrix[f"{i}-{j}"] != 10e6: # if the transition is valid
                    cost += transition_matrix[f"{i}-{j}"] # add the cost for this transition
                    if verbose: # print the tranformation of the objects
                        n1 = objects1[i]
                        n2 = objects2[j]
                        print (f"Tranform {n1.intersection(n2)} from {n1 - n2} -> {n2 - n1}")
                    # and remove the item from the list of items in each image
                    objects1.pop(i, None)
                    objects2.pop(j, None)

    for i, obj in objects1.items(): # any items remaining in instance 1 must be removed
        cost += len(obj)
        if verbose:
            print (f"Remove: {obj}")

    for i, obj in objects2.items(): # any items remaining in instance 1 must be added
        cost += len(obj)
        if verbose:
            print (f"Add: {obj}")

    if verbose:
        print ("--------------------------------------")
    return cost

def obj_distance(obj1, obj2):
    """"
    Τhe function of calculating the transition of each object to another.
    """
    lca = obj1.intersection(obj2)
    diffs = len(obj1 - lca) +  len(obj2 - lca)
    if diffs < 15:
        return diffs
    else:
        return 10e6
