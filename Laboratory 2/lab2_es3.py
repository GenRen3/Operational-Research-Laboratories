import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = [x**2 for x in range(3, 12)]
DELTA = 4


def traffic_matrix(nodes):

    tsd = np.random.uniform(0, 4, (nodes, nodes))
    np.fill_diagonal(tsd, 0)

# caso con high and low traffic    
#    tsd = np.zeros((nodes,nodes))
#    for s in range (0,nodes):
#        for d in range (0,nodes):
#            my_choice = np.random.randint(0,10)
#            if my_choice <= 1:
#                tsd[s,d] = np.random.uniform(0, 3) # low traffic
#            else:
#                tsd[s,d] = np.random.uniform(5, 15) # high traffic
#    np.fill_diagonal(tsd,0)
    
    return tsd

def plots(resg, resr):
    plt.figure()
    
    plt.plot(N,resg[0,:], 'bo-')
    
    plt.plot(N,resr[0,:], 'ko--')
    
    plt.xlabel('nodes')
    plt.legend(['greedy algorithm', 'random topology'])
    plt.ylabel(r'fmax')
    title = 'fmax vs nodes'
    plt.xticks(N)
    plt.title(title)
#    plt.margins(0.01, 0.1)
    plt.grid(which='both', axis='y')
#    plt.savefig(str(title) + '.png')
    plt.show()
    
class Topology(object):
    
    def __init__(self, tsd, nodes):
        self.tsd = tsd
        self.nodes = nodes
       
    def grid_dimensions(self):
        
        rad_nodes = np.sqrt(self.nodes)
        floor = int(np.floor(rad_nodes))
        ceil = int(np.ceil(rad_nodes))
    
        found = 0
        while (found == 0):
            if (self.nodes%floor == 0) and (floor >0):
                rows = floor
                found = 1
            elif (self.nodes%ceil == 0):
                rows = ceil
                found = 1
            else:
                floor-=1
                ceil+=1
        
        cols = int(self.nodes/rows)
        
        return (rows,cols)
    
    def bij_creation(self,manhattan_grid):
        
        (rows,cols) = np.shape(manhattan_grid)
        bij = np.zeros((self.nodes,self.nodes))
        
        for i in range (0,rows):
            for j in range (0,cols):
                x = int(manhattan_grid[i, j])
                bij[x,int(manhattan_grid[i,(j+1)%cols])] = 1
                bij[x,int(manhattan_grid[i,(j-1)])] = 1
                bij[x,int(manhattan_grid[i-1,j])] = 1
                bij[x,int(manhattan_grid[(i+1)%rows,j])] = 1
                            
        return bij
    
    def traffic_routing(self,bij):
        
        nodes = self.nodes     
        tsd = self.tsd
        
        fij = np.zeros((nodes, nodes))
        
        # Graph creation.
        elist = np.argwhere(bij)
        nlist = np.arange(self.nodes)
        
        G = nx.DiGraph()
        G.add_nodes_from(nlist)
        G.add_edges_from(elist)
    #    nx.draw(G)
        
        # Route the traffic to compute fij.
        for s in range (0, nodes):
            for d in range (0, nodes):
                
                try:
                    my_path = nx.shortest_path(G,source = s, target = d)
                except:
                    my_path = []
                
                for i in range (0,len(my_path)-1):
                    fij[my_path[i],my_path[i+1]] = fij[my_path[i],my_path[i+1]]+tsd[s,d]
        
    #    print ('Nodes: ', N, 'Edges: ', G.number_of_edges(),'Fmax: ', np.max(fij))
        return (np.max(fij))
    
class Random(Topology):
            
    def random_Manhattan(self):
        
        n = self.nodes
        (rows,cols) = self.grid_dimensions()
        results = []
        count = 0
        
        
        while (count < 97):
                    
            # Variables.
            bij = np.zeros((n, n))
            my_grid = np.zeros((rows,cols))
            fij = np.zeros((n, n))
            
            # Creation of manhattan-like random topology.
            possible_nodes = np.arange(0, n)
            
            for i in range (0,rows):
                for j in range (0,cols):
                    my_choice = np.random.choice(possible_nodes)
                    possible_nodes = np.delete(possible_nodes,np.where(possible_nodes==my_choice))
                    my_grid[i,j] = my_choice
                    
            bij = self.bij_creation(my_grid)
            
            # Graph creation.
            elist = np.argwhere(bij)
            G = nx.DiGraph()
            my_nodes = np.arange(0, n)
            G.add_nodes_from(my_nodes)
            G.add_edges_from(elist)
        #   nx.draw(G)
                
            # Routing.
            fij = self.traffic_routing(bij)
        
            count += 1
            results.append(np.max(fij))
      
        return (np.mean(results))
    
class Greedy(Topology):    
    
    def test_node(self, row, col, my_grid, tsd, node, possible_nodes):
        score = 0.0
        for i in range(0, np.size(tsd,0)):
            if i in my_grid:
                index = np.argwhere(my_grid==i)
                dist = float(self.distance_cells(row, col, index[0, 0], index[0, 1], np.size(my_grid, 0), np.size(my_grid, 1)))
                score += tsd[node, i]/dist
                score += tsd[i, node]/dist
               
        return score

    def best_node(self, row,col,my_grid,possible_nodes,tsd):
        score = 0.0
        best = -1
        for i in range(0, len(possible_nodes)):
            s = self.test_node(row, col, my_grid, tsd, possible_nodes[i], possible_nodes)
            if s > score:
                score = s
                best = possible_nodes[i]
                
        return (score, best)
        
    def adjacent_cells(self, my_grid,row,col):
        rows= np.size(my_grid,0)
        cols= np.size(my_grid,1)
        count = 0
        if (my_grid[row,(col+1)%cols] != -1):
            count+=1
        if (my_grid[row,(col-1)] != -1):
            count+=1
        if (my_grid[row-1,col] != -1):
            count+=1
        if (my_grid[(row+1)%rows,col] != -1):
            count+=1
        return count
            
    def best_cells(self, my_grid):
        
        my_list = []
        max_adjacent = 0
        
        for i in range (0,np.size(my_grid,0)):
            for j in range (0,np.size(my_grid,1)):
                if (my_grid[i,j] == -1):
                    count = self.adjacent_cells(my_grid,i,j)
                    if count > max_adjacent:
                        max_adjacent = count
                        my_list = []
                        my_list.append((i,j))
                    elif count == max_adjacent:
                        my_list.append((i,j))
        return my_list
        
    def distance_cells(self, y1,x1,y2,x2,rows,cols):
        
        distx = x2 - x1
        disty = y2 - y1
        dist2x = 122212
        dist2y = 122212
        
        if (np.abs(distx) > 1):
            if (distx > 0):
                dist2x = x1 + cols - x2
            else:
                dist2x = x2 + cols - x1
        
        if(np.abs(disty)>1):
            
            if (disty > 0):
                dist2y = y1 + rows - y2
            else:
                dist2y = y2 + rows - y1
                
        if (dist2x < np.abs(distx)):
            distx = dist2x
        else:
            distx = np.abs(distx)
            
        if (dist2y < np.abs(disty)):
            disty = dist2y
        else:
            disty = np.abs(disty)
                
        return distx + disty

    def greedy_Manhattan(self):
        nodes = self.nodes
        tsd = self.tsd
        (rows,cols) = self.grid_dimensions()
    
        my_grid = np.ones((rows,cols))*(-1)
        
        #creation of manhattan-like random topology
        
        # il nodo che complessivamente scambia più traffico con tutti gli altri in ingresso e uscita
        starting_node = np.argmax(np.sum(tsd,0) + np.sum(tsd,1))
        my_grid[0,0] = starting_node
        possible_nodes = np.arange(0, nodes)
        possible_nodes = np.delete(possible_nodes,np.where(possible_nodes==starting_node))
              
        while (len(possible_nodes > 0)):
            b_cells = prova.best_cells(my_grid)
            b_cell = (-1, -1)
            b_node = -1
            b_score = 0.0
            for i in range(0, len(b_cells)):
                (s, n) = prova.best_node(b_cells[i][0], b_cells[i][1], my_grid, possible_nodes, tsd)
                if s > b_score:
                    b_cell = b_cells[i]
                    b_node = n
                    b_score = s
                
            my_grid[b_cell[0], b_cell[1]] = b_node
            possible_nodes = np.delete(possible_nodes,np.where(possible_nodes==b_node))

        bij = self.bij_creation(my_grid)
            
        #graph creation
        elist = np.argwhere(bij)
        G = nx.DiGraph()
        my_nodes = np.arange(0,nodes)
        G.add_nodes_from(my_nodes)
        G.add_edges_from(elist)
    #   nx.draw(G)
            
        #routing
        fij = self.traffic_routing(bij)
      
        return (np.max(fij))

if __name__ == '__main__':
 
    results_greedy = np.zeros((1,len(N)))   
    results_random = np.zeros((1,len(N)))
          
    for cnt, n in enumerate(N):
        tsd = traffic_matrix(n)
        my_greedy = Greedy(tsd, n)
        my_random = Random(tsd, n)

        results_greedy[0, cnt] = my_greedy.greedy_Manhattan()
        results_random[0, cnt] = my_random.random_Manhattan()
        
    plots(results_greedy, results_random)
