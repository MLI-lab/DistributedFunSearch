"""Extracted priority functions that achieved target signature."""

# Function 1
# Priority hash: e5886b77dc870bae
# Duplicate count: 14
# Source: checkpoint_2025-12-02_11-34-03.pkl
# Island: 9, Score: 172.0
# Scores: {(6, 1, 2): 10, (7, 1, 2): 16, (9, 1, 2): 52, (8, 1, 2): 30, (10, 1, 2): 94, (11, 1, 2): 172}
def priority_1(node, n, s, q):
    #Same logic applied here but instead we made it more dynamic   
        max_k=min([n//2 , n-s]) # Dynamic Maximum K limit
        sequence_score=[0]*(max_k+1) # store each score per possible k    

        for leng in reversed(range(1,max_k + 1)): # this loops over whole length n 
        
            scores = []

            for ind in range(n - leng + 1):

                current_subseq = node[ind : ind + leng]
                num_of_zeros = current_subseq.count("0")
            
                weight = pow(leng * num_of_zeros / (leng + num_of_zeros), .68 )            
                scores.append(-weight*num_of_zeros/(n-s)*n*0.9765625)
            
            total_scores=sum(scores)/pow(leng,.3)        
            sequence_score[leng]=total_scores
        
        tssum=sum(sequence_score[:])    
        return (((tssum))*np.log(max_k+1))



