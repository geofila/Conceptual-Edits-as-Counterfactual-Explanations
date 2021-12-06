import numpy as np
from nltk.corpus import wordnet as wn
from Queries import *

def create_tbox_coco(queries_coco,materialize = False):
    ## A set of all concpets which appear
    all_concepts = set()
    for q in queries_coco:
        for s in queries_coco[q].concepts:
            all_concepts = all_concepts.union(s)

    ## the tbox (dictionary: key is concept, value is definition (list of concepts))
    tbox = {c:set() for c in all_concepts}
    ## connect with wordnet
    for c in tbox:
        syns = wn.synsets(c)
        if len(syns)>0:
            tbox[c].add(wn.synsets(c)[0].name())
        else:
            syns = wn.synsets(c.replace(' ',''))
            if len(syns)>0:
                tbox[c].add(wn.synsets(c.replace(' ',''))[0].name())
            else:
                wrds = c.split(' ')
                for w in wrds:
                    if len(wn.synsets(w))>0:
                        tbox[c].add(wn.synsets(w)[0].name())

    if materialize:
        for c in tbox:
            while True:
                curr_len = len(tbox[c])
                new_syns = set()
                for syn in tbox[c]:
                    hypers = [c.name() for c in wn.synset(syn).hypernyms()]
                    new_syns = new_syns.union(hypers)
                tbox[c] = tbox[c].union(new_syns)
                if len(tbox[c])==curr_len:
                    break
    return tbox

def rewrite_query(q,tbox):
    new_query = Query()
    for c in q.concepts:
        new_c = set()
        for cc in c:
            new_c.add(cc)
            for ccc in tbox[cc]:
                new_c.add(ccc)
        new_query.concepts=np.append(new_query.concepts,new_c)
    return new_query
