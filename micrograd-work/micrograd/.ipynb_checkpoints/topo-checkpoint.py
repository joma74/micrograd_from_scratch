from micrograd.engine import Value

def build(rootTerm: Value):
    assert isinstance(rootTerm, Value), "term must be instance of type Value"
    # topological order all of the terms in the graph
    topo = []
    visited = set()
    return _build_recurs(rootTerm, topo, visited)

def _build_recurs(term: Value, topo: [Value], visited: [Value]):
    # case when a term is used more than once in teh graph
    if term not in visited:
        visited.add(term)
        for unvisited_term in term._terms:
            _build_recurs(unvisited_term, topo, visited)
        topo.append(term)
    return topo

def findLeafNodes(topo: [Value]):
    assert isinstance(topo, list), "topo must be instance of type list"
    leafNodes = []
    for term in reversed(topo):
        if(not len(term._terms)): leafNodes.append(term)            
    return leafNodes
            