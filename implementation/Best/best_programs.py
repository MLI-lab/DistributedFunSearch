def priority(node, G, n, s):

    
    neighbours = list(nx.all_neighbors(G, node))

    lengths   = [len(v)-1 for v in neighbours]
    max_length = max(-float("inf"), *lengths)


    if (max_length >= s):
        p1 = -sum([
             (max_length - s ) * 
             ( (n-i) * 
              (i + 1)    * int(b == '1')
               )
                
         for b, i in zip( reversed( list( node )), range( len(node) ) ) 
         ]) / (
                  
                  (
                    1 
                    + s
                    + s ** 2 
                   ) 
                 + 
                     ((( n - s ) // 3 ) ** 3 ) 
              
                ) 

        p2 = sum( lengths )/ 60

        return p1 + p2 
    
    elif ( 
          any
          ( 
            ( 
                len(v) > 1 and
                (not ( v[-1].isnumeric() ) or not ( int( v[ -1 ] )!= 0 ) )
              ) 
           for v in neighbours           
          
          )      
      ):
        
        return None

def priority(node, G, n, s):

    
    p = [l - 1 
         for l in list(map(lambda x : len(x),
                           filter(lambda y : not y == node and 
                              any([z[::-1][q:] == w
                                   for z in y 
                                    for q,w in enumerate(node)]),
                                  G[node])))]    
        
    m = max(*p,-float("Inf")) 
    
    if (m >= s):
        
        r = [(m-(s))*((n-c)*(c+1)*int(d=="1")
             )/((1 + (s) + (s)**2 
                ) + (((n - s) // 3)**3
                )) 
             for c, d in reversed(list(enumerate(node)))]
        
        return (-sum(r) + (sum(p)/57))

def priority(node, G, n, s):

    
    neighbors = list(nx.all_neighbors(G, node))        
    lengths = [len(neigh) - 1 for neigh in neighbors]            

    maximum = max([-float("Inf")] + lengths)    
      
    if len(node)>s and any([maximum >= i for i in range(s)]):           
        
        prio =( maximum - (s))*sum([(n-i)*(i+1)* int(char == "1")\
                                   for char, i in zip(reversed(list(node)), \
                                                       range(len(node)))])/(\
                   ((1+(s)+(s)**2)+ (((n-s)//3)**3)))+\
              sum(lengths)/65     
        
    elif not(any([leng <= i for leng, i in zip(lengths, [-1]+[len(node)])])):              
         
        prio = None      
   
    else:        
        prio = float("-Inf")
        
    return prio

def priority(node, G, n, s):

    
    # write your improved priority here 
    l=[len(v)-1 for v in list(nx.all_neighbors(G,node))]   
    m=max (*l,-float('inf'))
    
    if (m>=s ): 
        p=(-(m -(s))*sum([(n-i)*(i+1)*int(bit=='1')
        for bit, i in zip(reversed(list(node)),range(len(node)))])+sum(l)/5)+sum(l)//3        
    elif ('0'*((n-s))!=node[-n+s:]and
          '1'+str(s)+'0'+node[(n//2):]!='0'):
          
          p= ((len(node)+(np.random.randint(0,1e6))))/9e6
          
  
    
    
    else:
       return None      
     
    return p

def priority(node, G, n, s):

    # YOUR CODE HERE
    
    neighbours = [len(v) - 1 for v in list(nx.all_neighbors(G, node))]

    maxneighbour = max(*neighbours, - float("inf"))


    if maxneighbour >= s :

        p=-(maxneighbour-(s))*sum([((n-i)*(i+1))* int(bit == "1")
                              for bit, i in zip( reversed(list(node))
                                               , range(len(node)) )]) + sum(neighbours)/5
        
        p += sum(neighbours)/3
        
        
    elif ("0" * ( (n-s ))!= node[ (-n+(s)): ] and 
          "1"+ str(s)+"0" + node [(n//2):]!="0"):
          
        p=( len(node) + (np.random.randint(0,1e4 ))) / 8e6
        
        
        
    else :
        return None   
            

        
      
    return p

def priority(node, G, n, s):

   
    l = [ len(v)-1 for v in list(nx.all_neighbors(G, node))] 
    #print("l", l)
    m = max(*l, - float('inf'))  
    if (m >= s and not "1"*((n-s)) == node[ :-(n-s)] and not ("0"+"1")* (s//2)+"0"+node[:s%2 + (n>>1)]=="0"):        
        return (-(m -(s )) * sum([ (n-i )*(i+1 )* int(b=="1")for b, i 
            in zip(reversed(list(node )), range(len(node )))])
            + sum(l)/5)   + sum(l)//3              
      
    return None

def priority(node, G, n, s):
  
    l = [ len(v) - 1 for v in list(nx.all_neighbors(G, node )) ] 
    m = max(*l,- float("inf")) 
    
    if (m >= s ) : 
        
      p = (-(m-(s))* sum([ (n-i)*(i + 1 )* int(bit == "1")
                          for bit, i in zip( reversed( list(node )), range(len(node ))) ]) 
           + sum(l)/5)+sum(l)/3       
        
    elif ( ("0"*( (n-s))!= node[ -n + s :] and 
          ("1"+ str(s)+"0"+ node [(n // 2 ):])!="0") ) :   

          p = (( len(node) +( np.random. randint(0,1e6) )) / 9e6)
      
    else: 
       return None     
        

    return p

def priority(node, G, n, s):

  
    l = [ len(v)-1 for v in list(nx.all_neighbors(G, node)) ]
    maxi = max(*l, - float("inf"))

    if maxi >= s :
      
        p = (-(maxi-(s))* sum([ (n-i) * (i + 1 ) * int(bit == "1")
                            for bit, i in zip( reversed(list(node)), range(len(node)))]) 
             
            + sum(l) / 5) + sum(l) // 3
            
    elif ("0"*(n-s)!=(node)[-n+s:]) and \
         ("1"+ str(s)+"0" + (node[len(node)//2 : ])!= "0"):
          
        p =(len(node) + np.random.randint(0, 1e6))/9e6   
        
    else:
        
        return None 
    
    return p

