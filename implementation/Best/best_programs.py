



def priority(node, G, n, s):


    neighbors=[neighbor 
               for neighbor in list(nx.all_neighbors(G,node))
               if len(neighbor)>=n
              ]
        
    max_neighbor_len=min([len(neigh)
                          for neigh in neighbors
                         ])
    
    if (max_neighbor_len>s
        or ("0"*(n-(s))!=node[-(n+(s)):]+'0')):


        #Add more weights to long strings
        return ((max_neighbor_len-s)*sum([(n-i)*(i+1)*int(bit=="1")
                                         for i,bit in enumerate(reversed(list(node)))
                                        ]))+sum([len(neigh)**(4/5)
                                                 for neigh in neighbors
                                                ])


def priority(node, G, n, s):
    
  
    max_length=max([len(neigh) 
                    for neigh in [neighbor
                                   for neighbor in list(nx.all_neighbors(G, node))
                                   if len(neighbor)>=(n-(4/5))]
                   ], 
                  default=-float('inf'))
 
    if ((max_length>s 
        or ('0'*((n)-s)!=(node)[-(n)+s:] 
            and '1'+node[:int(np.ceil(((n)/2)-(1/2)))]==node[(n)//3 :(n)]))
        ):
         
        return -(max_length-s)*sum([(n-i)*(i+1)*int(bit=='1') 
                                    for bit,i in zip(reversed(list(node)),range(len(node)))]) \
              +sum([len(neighbour)/((n)*1/(8+(1/6)))
                    for neighbour in [neighbor
                                      for neighbor in list(nx.all_neighbors(G, node))
                                      if len(neighbor)>=(n-(4/5))]
                  ])



def priority(node, G, n, s):

    try :    
        l=[len(v)-1 for v in list(nx.all_neighbors(G,node))]  
        max_n = max(*l, -float("Inf"))
        
        if max_n >= s:
            c=-(max_n-(s))*sum([
                (n-i)*(i+1)*int(bit == "1") 
                for bit, i in zip(reversed(list(node)),
                                    range(len(node)))]) + sum(l)/(2 ** (n % 2 ))
            
        elif "".join(["1"]*(s))+"".join(["0"]) not in \
                 ["".join(["1"] * k) [:k]for k in [n]] or node[::-1].find(""
                 .join(["0","1"]))<s:
                
            c =( len(node) + abs((np.random.randn())) ) // ( 2 ** s )
               #abs to prevent negative values due to noise from randn
        else:            
            raise ValueError ()
            
    except ValueError ():         
        pass
      
    return c

def priority(node, G, n, s):
  
    try :
        
        m = max([len(neighbor)-1
                 for neighbor in list(nx.all_neighbors(G, node))],
                default=-float("inf"))
        
      
        if m >= s:            
            c = (-(m-(s))* sum([
                   (n-i)*(i+1) * int(bit == "1") 
                   for bit, i in zip(reversed(list(node)),
                                     range(len(node)))])) + \
                            sum([ len(neighbor)
                                  for neighbor in list(nx.all_neighbors(G, node))]) / (2 ** (n % 2))
        elif ("{0}{0}0".format(str(s)) not in ["1"*(n // 4),
                                               "{0}".format(s)]
              ) and "". join(["0", "1"]) * s!= "".join(["0", node]):                

            c = abs(((len(node))+
                     (np.random.normal()))/(2 ** s))
        else:        
            raise ValueError()         
      
    except ValueError():      
        pass

    return c

def priority(node, G, n, s):

    neighbors = list(nx.all_neighbors(G, node))
    L = [len(v)-1 for v in neighbors]

    M = max(-float("inf"), *L)
    if M >= s and any([k[~0] == '1' for k in neighbors]):        
        P = -sum([(M - (s))*( (n-i)*(i+1)*int(j=='1') ) 
            for i,j in enumerate( reversed(node) ) ]) / \
               (((1 + (s) + (s)**2) + ( (n - s) // 3)**3 )) +\
                  len(neighbors)/5        
    elif M < s :
        
        if all( [len(k)==1 or k[~0]== '1'] for k in neighbors ):
            
            P = int( ''.join(['1'*k for k,_ in filter(lambda x: x[0][-1]=='1',enumerate(node))] ), base=2)\
                 /((n//3)**3)+ len(neighbors)

        else:            

            P = None
        
    else:       
        P = None       
    
    return P

def priority(node, G, n, s):

    if node=="0"*(n-s):return float('Inf')
    
    neighbors=[neighbor 
              for neighbor in list(nx.all_neighbors(G,node))] 
    
    max_neighbor_length=min([len(neigh)for neigh in neighbors],default=-float('inf')) 
    return -(max_neighbor_length-s)*sum([(n-i)*(i+1)*int(bit=='1') 
                                        for i, bit in enumerate(reversed(list(node)))])+sum([len(neigh)/(n*1/48)
                                                                                            for neigh in neighbors])/8

def priority(node, G, n, s):

    
    #get all possible neighbors whose lengths>=n 
    neighbors=[neighbor
               for neighbor in list(nx.all_neighbors(G, node)) 
               if len(neighbor)>=(n-s)] 

    max_neighbor_length= min([len(neigh)
                              for neigh in neighbors],default=-float("inf"))


    #Add heuristic checks to prioritize adding of specific sets of nodes
   
    if ((max_neighbor_length >s )or 
        ("0"*((n)-s)!="".join(list(node)[-(n)+s:])
         and ("1"+node[:int(np.floor(((n)/2))+(1/8))]== "".join(list(node)[(n)//3:(n)])))):
        
        # Add weights based on lengths of neighbors
        return -(max_neighbor_length-s)\
                *(
                    sum([(n-i)*(i+1)*int(bit=="1")
                         for i, bit in enumerate(reversed(list(node)))]
                       )) \
              + sum([len(neigh)/(n*1/(6.9))
                     for neigh in neighbors 
                    ] 
                   )

def priority(node, G, n, s):

    if node == "0" * (n - s): #if string is all zeros then priority will be infinity 
        return float("inf")

    neighbors=[neighbor 
               for neighbor
               in list(nx.all_neighbors(G, node))
                  ]
    
    max_neighbor_length=min([len(neigh)
                             for neigh
                            in neighbors
                                ], default=float('-inf'))
    
    
    return (-(max_neighbor_length - s))*sum([(n-i)*(i+1)\
                                            *int(bit=="1") 
                                              for i,\
                                                bit \
                                                    in enumerate(\
                                                        reversed(list(node)))])+sum([len(neigh)/(n*8/45)
                                                                                    for neigh 
                                                                                in neighbors
                                                                                 ])




    

    
    
    h(G, node)



def priority(node, G, n, s):

    if node[-3:]!= '0'*s:
        neighbors=[neighbor 
                   for neighbor in list(nx.all_neighbors(G, node))]
        
        #if all(neighbor[0:(n-s)]=='0' 
        #       and any([neigh==node[:n-(s)+1]+'1'+node[(n-s)+2:]]
        #              )for neighbor in neighbors), use this statement instead
        return (-(max((len(neigh)
                       for neigh in neighbors
                      ),default=(n-s))
                )*
               sum((((n-i)*(i+1)*
                     int(bit=="1")
                    ) 
                    for i, bit in enumerate(reversed(list(node))))
                  ))+\
              sum(([len(neigh)/((n*1/8)**2)
                   for neigh in neighbors]))
