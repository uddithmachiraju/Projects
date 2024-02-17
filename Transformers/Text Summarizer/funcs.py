import torch, math 
import torch.nn.functional as F 

def scalar_dot_product(key, query, value, mask = None):
    query_size = query.size()[-1] 
    # Permute the key dimentions for matrix multiplications 
    # (batch_size, heads, max_length, head_division) --> (batch_size, heads, head_divisions, max_length) 
    # The resultant score matrix will be of shape (bacth_size, heads, max_length, max_length)
    # Dividing with query_size will normalize the score vector 

    scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query_size) 
    
    if mask is not None:
        scores += mask 

    # Apply softmax on scores 
    weights = F.softmax(scores, dim = -1) 
    weights = F.dropout(weights, 0.1) 

    # (batch_size, heads, max_length, max_length) --> (batch_size, heads, max_length, head_divisions) 
    # (batch_size, heads, max_length, head_divisions) 
    values = torch.matmul(weights, value) 

    return weights, values  

def positional_embeddings():
    pass 