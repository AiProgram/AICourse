import networkx
import heapq
import getData
import queue

def hFun(city,graph):
    return graph.nodes[city]["goalDist"]

def aStarSearch(graph,startCity):
    graph.graph["start"]=startCity
    openList=[]
    closeList=[]
    pathRecord={}
    openList.append((0+hFun(startCity,graph),0,startCity))
    while len(openList)>0:
        thisCity=heapq.heappop(openList)
        if thisCity[2]==graph.graph["goal"]:
            print(thisCity[1])
            break
        else:
            print(thisCity[2])
        closeList.append(thisCity[2])
        for nCity in graph[thisCity[2]]:
            costed=thisCity[1]+graph[thisCity[2]][nCity]["dist"]
            heapq.heappush(openList,(costed+hFun(nCity,graph),costed,nCity))

def BFS(graph,startCity):
    graph.graph["start"]=startCity
    q=queue.Queue()
    q.put((startCity,0))
    visited=[]
    while q.qsize()>0:
        thisCityInfo=q.get()
        if thisCityInfo[0] == graph.graph["goal"]:
            print(thisCityInfo[1])
            break
        if thisCityInfo[0] not in visited:
            visited.append(thisCityInfo[0])
            for nCity in graph[thisCityInfo[0]]:
                q.put((nCity,thisCityInfo[1]+graph[thisCityInfo[0]][nCity]["dist"]))

def UCS(graph,startCity):
    graph.graph["start"]=startCity
    q=[]
    visited=[]
    q.append((0,startCity))
    while len(q)>0:
        thisCityInfo=heapq.heappop(q)
        if thisCityInfo[1]==graph.graph["goal"]:
            print(thisCityInfo[1]+" "+str(thisCityInfo[0]))
            break
        if thisCityInfo[1] not in visited:
            visited.append(thisCityInfo[1])
            for nCity in graph[thisCityInfo[1]]:
                heapq.heappush(q,(thisCityInfo[0]+graph[thisCityInfo[1]][nCity]["dist"],nCity))

def subDFS(graph,visited,thisCity,cost):
    if thisCity==graph.graph["goal"]:
        print(thisCity+" "+str(cost))
        return True
    if thisCity not in visited:
        visited.append(thisCity)
        for nCity in graph[thisCity]:
            if subDFS(graph,visited,nCity,cost+graph[thisCity][nCity]["dist"]):
                print(thisCity)
                return True
    return False
    

def DFS(graph,startCity):
    graph.graph["start"]=startCity
    visited=[]
    subDFS(graph,visited,startCity,0)

if __name__=="__main__":
    #aStarSearch(getData.readGraph(),"Arad")
    #BFS(getData.readGraph(),"Arad")
    #UCS(getData.readGraph(),"Arad")
    DFS(getData.readGraph(),"Arad")