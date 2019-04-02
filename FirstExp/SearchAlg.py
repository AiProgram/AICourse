import networkx
import heapq
import getData
import queue
import sys
def hFun(city,graph):
    return graph.nodes[city]["goalDist"]

def obtainPath(comeFrom,graph):
    cost=0
    path=[]
    current=graph.graph["goal"]
    while True:
        path.append(current)
        if current==graph.graph["start"]:
            break
        pre=comeFrom[current]
        cost+=graph[pre][current]["dist"]
        current=pre
    return path,cost

def aStarSearch(graph,startCity):
    graph.graph["start"]=startCity
    closedSet=[]
    openSet=[startCity]
    comeFrom={}
    gScore={}
    fScore={}
    for node in graph.nodes():
        gScore[node]=sys.maxsize
        fScore[node]=sys.maxsize
    gScore[startCity]=0
    fScore[startCity]=hFun(startCity,graph)
    while len(openSet)>0:
        current=openSet[0]
        for city in openSet:
            if fScore[city]<fScore[current]:
                current=city
        if current== graph.graph["goal"]:
            pass
        openSet.remove(current)
        closedSet.append(current)
        for nCity in graph[current]:
            if nCity in closedSet:
                continue
            newScore=gScore[current]+graph[current][nCity]["dist"]
            if nCity not in openSet:
                openSet.append(nCity)
            elif newScore>=gScore[nCity]:
                continue
            comeFrom[nCity]=current
            gScore[nCity]=newScore
            fScore[nCity]=gScore[nCity]+hFun(nCity,graph)
    return obtainPath(comeFrom,graph)


def BFS(graph,startCity):
    graph.graph["start"]=startCity
    q=queue.Queue()
    q.put((startCity,0))
    visited=[]
    comeFrom={}
    while q.qsize()>0:
        thisCityInfo=q.get()
        if thisCityInfo[0] == graph.graph["goal"]:
            break
        if thisCityInfo[0] not in visited:
            visited.append(thisCityInfo[0])
            for nCity in graph[thisCityInfo[0]]:
                q.put((nCity,thisCityInfo[1]+graph[thisCityInfo[0]][nCity]["dist"]))
                if nCity not in visited:
                    comeFrom[nCity]=thisCityInfo[0]
    return obtainPath(comeFrom,graph)

def UCS(graph,startCity):
    graph.graph["start"]=startCity
    q=[]
    visited=[]
    opened=[]
    comeFrom={}
    q.append((0,startCity))
    opened.append(startCity)
    while len(q)>0:
        thisCityInfo=heapq.heappop(q)
        opened.remove(thisCityInfo[1])
        if thisCityInfo[1]==graph.graph["goal"]:
            break
        if thisCityInfo[1] not in visited:
            visited.append(thisCityInfo[1])
            for nCity in graph[thisCityInfo[1]]:
                if nCity not in opened or nCity not in visited:
                    heapq.heappush(q,(thisCityInfo[0]+graph[thisCityInfo[1]][nCity]["dist"],nCity))
                    opened.append(nCity)
                    if nCity not in visited:
                        comeFrom[nCity]=thisCityInfo[1]
                elif nCity in opened:
                    newCost=thisCityInfo[0]+graph[thisCityInfo[1]][nCity]["dist"]
                    for cityInfo in q:
                        if cityInfo[1]==nCity:
                            if cityInfo[0]>newCost:
                                q.remove(cityInfo)
                                q.append((newCost,nCity))
                                heapq.heapify(q)
                                comeFrom[nCity]=thisCityInfo[1]
                        break
    return obtainPath(comeFrom,graph)

def subDFS(graph,visited,thisCity,cost,comeFrom):
    if thisCity==graph.graph["goal"]:
        return True
    if thisCity not in visited:
        visited.append(thisCity)
        for nCity in graph[thisCity]:
            if nCity not in visited:
                comeFrom[nCity]=thisCity
            if subDFS(graph,visited,nCity,cost+graph[thisCity][nCity]["dist"],comeFrom):
                return True
    return False
    

def DFS(graph,startCity):
    graph.graph["start"]=startCity
    visited=[]
    comeFrom={}
    subDFS(graph,visited,startCity,0,comeFrom)
    return obtainPath(comeFrom,graph)

def subHillClimbing(graph,thisCity,visited,cost,comeFrom):
    if thisCity==graph.graph["goal"]:
        return True
    if thisCity not in visited:
        visited.append(thisCity)
        visitList=[(graph.nodes[nCity]["goalDist"],nCity) for nCity in graph[thisCity]]
        heapq.heapify(visitList)
        while len(visitList)>0:
            nextCityInfo=heapq.heappop(visitList)
            nextCity=nextCityInfo[1]
            if nextCity not in visited:
                comeFrom[nextCity]=thisCity
            if subHillClimbing(graph,nextCity,visited,cost+graph[thisCity][nextCity]["dist"],comeFrom):
                return True
    return False

def hillClimbing(graph,startCity):
    graph.graph["start"]=startCity
    visited=[]
    comeFrom={}
    subHillClimbing(graph,startCity,visited,0,comeFrom)
    return obtainPath(comeFrom,graph)

if __name__=="__main__":
    startCity="Timisoara"
    path,cost=aStarSearch(getData.readGraph(),startCity)
    path.reverse()
    print(str(path)+" "+str(cost))
    print("------------------------------")

    path,cost=BFS(getData.readGraph(),startCity)
    path.reverse()
    print(str(path)+" "+str(cost))
    print("------------------------------")

    path,cost=UCS(getData.readGraph(),startCity)
    path.reverse()
    print(str(path)+" "+str(cost))
    print("------------------------------")

    path,cost=DFS(getData.readGraph(),startCity)
    path.reverse()
    print(str(path)+" "+str(cost))
    print("------------------------------")

    path,cost=hillClimbing(getData.readGraph(),startCity)
    path.reverse()
    print(str(path)+" "+str(cost))
    print("------------------------------")