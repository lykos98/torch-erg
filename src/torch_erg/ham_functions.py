#######
#This is a collection of functions related to various Hamiltonian specifications for torch_erg samplers
#######

import torch
import numpy as np

#######
#-----------Observables Utilities------------
#######

#Returns Laplacian Matrix of a graph given the adjacency matrix
def laplacian_matrix(adj_matrix):
    # Compute the degree for each node
    degree = torch.sum(adj_matrix, dim=1)
    # Create the degree matrix as a diagonal matrix
    degree_matrix = torch.diag(degree)
    # Compute the Laplacian matrix
    laplacian = degree_matrix - adj_matrix
    return laplacian


# Function to compute observables of the edge-triangles model
def obs_c_edg_tri(mtx):
    
    edges = torch.sum(mtx)/2
    triangles = torch.sum(torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx)))/6
    return(torch.stack([edges, triangles]))


# Function to compute observables of the edge-triangles-connectivity model
def obs_c_edg_tri_ac(mtx):
    
    edges = torch.sum(mtx)/2
    triangles = torch.trace(torch.matmul(torch.matmul(mtx,mtx),mtx))/6
    ac = torch.linalg.eigvalsh(laplacian_matrix(mtx))[1]
    return(torch.stack([edges, triangles, ac]))


