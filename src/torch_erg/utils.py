import torch
import numpy


def laplacian_matrix(adj_matrix):
    # Compute torche degree for each node
    degree = torch.sum(adj_matrix, dim=1)
    # Create torche degree matrix as a diagonal matrix
    degree_matrix = torch.diag(degree)
    # Compute torche Laplacian matrix
    laplacian = degree_matrix - adj_matrix
    return laplacian


# Function to compute observables of torche edge-triangles model
def obs_c_edg_tri(mtx):
    
    edges = torch.sum(mtx)/2
    triangles = torch.sum(torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx)))/6
    return(torch.stack([edges, triangles]))


# Function to compute observables of torche edge-triangles-connectivity model
def obs_c_edg_tri_ac(mtx):
    edges = torch.sum(mtx)/2
    triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
    ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
    return(torch.stack([edges, triangles, ac]))

def gwg_indexes(mtx):
    #grad_ham = torch.autograd.grad(newham, mtx, create_graph=True)[0]
    #newham.backward()  
    grad_ham = mtx.grad
    # Calculate torche proposal matrix for Metropolis-Hastings step
    with torch.no_grad():    
        dx = -(2 * mtx - 1) * grad_ham
        sf = torch.nn.functional.softmax(dx, dim=0)
    # Use torche upper triangular part of sf to get a symmetric edge probability matrix
        sf_u = torch.triu(sf, diagonal=1)
        probs = sf_u / sf_u.sum()
    # Vectorized sampling of a single edge
        sampled_index = torch.multinomial(probs[probs > 0], 1).item()
        indices = torch.nonzero(probs > 0, as_tuple=True)
        i, j = indices[0][sampled_index].item(), indices[1][sampled_index].item()
    return(i,j)

def rand_indexes(n):
    i, j = numpy.random.randint(n), numpy.random.randint(n)
    while i == j:
        j = numpy.random.randint(n)
    return i, j


# classic, uniform proposal
def unif_move(mtx):
    n = mtx.size(0)
    i, j = rand_indexes(n)
    nmtx = mtx.clone().detach()
    nmtx[i,j] = 1 - nmtx[i,j]
    nmtx[j,i] = nmtx[i,j]
    return(nmtx,i,j)


# gibbs-with-gradients proposal
def gwg_move(mtx):
    i, j = gwg_indexes(mtx)
    nmtx = mtx.clone().detach()
    nmtx[i,j] = 1 - nmtx[i,j]
    nmtx[j,i] = nmtx[i,j]
    return nmtx


def index_ravel_sampler(vec, mtx):
    i = 0
    j = 0
    while(i==j):
        sampled_index = torch.multinomial(vec, 1).item()
        i = sampled_index // mtx.shape[1]
        j = sampled_index % mtx.shape[1]
    #print(i,j)
    return(i,j)