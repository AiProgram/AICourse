import csv
import networkx as nx
import os

def readGraph():
    thisFolder=os.path.abspath(os.path.dirname(__file__))
    graph=nx.Graph()
    with open(thisFolder+"\data\graph.csv") as graphFile:
        reader=csv.DictReader(graphFile)
        for row in reader:
            graph.add_edge(row['from'],row['to'],dist=int(row['distance']))
    with open(thisFolder+"\data\cityDistance.csv") as distFile:
        reader=csv.DictReader(distFile)
        for row in reader:
            graph.add_node(row['city'],goalDist=int(row['distance']))
    graph.graph['goal']="Bucharest"
    return graph
    

if __name__=="__main__":
    readGraph()
    pass