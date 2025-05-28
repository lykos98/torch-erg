import torch
import src.torch_erg.betas_compute as bc 



def compute_k_triangle(mtx):
    # Ensure the tensor is in COO format for efficient processing
    if not mtx.is_sparse:
        mtx = mtx.to_sparse()

    # Get the indices of non-zero elements (connections)
    indices = mtx.coalesce().indices()  # Shape: [2, num_non_zero]
    
    tricount = 0
    tricount2 = 0
    
    # For each pair (i, j) where there is an edge (mtx[i, j] == 1)
    for idx in range(indices.size(1)):
        i, j = indices[:, idx]
        
        # Find common neighbors of i and j to count triangles
        common_neighbors = (mtx[i].to_dense() * mtx[j].to_dense()).sum().item()
        
        if common_neighbors >= 2:
            tricount2 += common_neighbors * (common_neighbors - 1) / 2
        tricount += common_neighbors

    return int(tricount / 6), int(tricount2 / 2)

def compute_2_triangle(adj):
    # Using matrix multiplication to count the number of 2-triangles
    A = adj.to(torch.float32)  # Ensure A is a float tensor
    A_squared = torch.mm(A, A)  # A^2 (squared adjacency matrix)

    # Count the number of 2-triangles
    two_triangle_count = torch.sum(A_squared) - torch.trace(A_squared)  # Exclude self-loops
    return int(two_triangle_count / 2)  # Each 2-triangle is counted twice

def linktype(a, b):
    return (a * b - 1) if (a + b < 5) else (a + b - 1)

def deg_distr(adj):
    return torch.sum(adj, dim=1).numpy()  # Sum along rows to get degree distribution

def check_existing_triangle(i, k, adj):
    return torch.sum(adj[k] * adj[:, i]).item()

def change_triang3(adj, i, j):
    count = 0
    kcount = 0
    lcount = 0
    rcount = 0
    n = adj.shape[0]
    
    neighbors_i = adj[i]  # Get neighbors of node i
    neighbors_j = adj[j]  # Get neighbors of node j

    # Count triangles and potential triangle connections
    for k in range(n):
        if neighbors_i[k] == 1 and neighbors_j[k] == 1:
            count += 1
            lcount += check_existing_triangle(i, k, adj) - adj[i, j]
            rcount += check_existing_triangle(j, k, adj) - adj[i, j]

    kcount = count * (count - 1) / 2 if count >= 2 else 0
    return count, (kcount + lcount + rcount)



#-------------obs computation methods--------------------------


def obs_edg_tri_2tri(adj, q1, q2, q3):
    # Calculate edges and triangles for different regions
    e_gg = torch.sum(adj[0:q1, 0:q1]) / 2
    e_ll = torch.sum(adj[q1:(q1 + q2), q1:(q1 + q2)]) / 2
    e_ii = torch.sum(adj[(q1 + q2):, (q1 + q2):]) / 2
    e_gl = torch.sum(adj[0:q1, q1:(q1 + q2)])
    e_gi = torch.sum(adj[0:q1, (q1 + q2):])
    e_li = torch.sum(adj[q1:(q1 + q2), (q1 + q2):])
    t, tk = compute_k_triangle(adj)
    
    return torch.tensor([e_gg, e_gl, e_gi, e_ll, e_li, e_ii, t, tk])

def obs_er_tri_2tri(adj):
    e = torch.sum(adj) / 2
    t, tk = compute_k_triangle(adj)
    return torch.tensor([e, t, tk])

def obs_edg(adj, q1, q2, q3):
    e_gg = torch.sum(adj[0:q1, 0:q1]) / 2
    e_ll = torch.sum(adj[q1:(q1 + q2), q1:(q1 + q2)]) / 2
    e_ii = torch.sum(adj[(q1 + q2):, (q1 + q2):]) / 2
    e_gl = torch.sum(adj[0:q1, q1:(q1 + q2)])
    e_gi = torch.sum(adj[0:q1, (q1 + q2):])
    e_li = torch.sum(adj[q1:(q1 + q2), (q1 + q2):])
    
    return torch.tensor([e_gg, e_gl, e_gi, e_ll, e_li, e_ii])

def obs_avgdeg(adj, q1, q2, q3):
    d1 = torch.sum(adj[0:q1]) / q1
    d2 = torch.sum(adj[q1:q1 + q2]) / q2
    d3 = torch.sum(adj[q1 + q2:]) / q3
    return d1.item(), d2.item(), d3.item()

def obs_edg_ddeg(adj, q1, q2, q3):
    obsD = deg_distr(adj)
    obsE = obs_edg(adj, q1, q2, q3)
    obs = torch.cat((obsD, obsE))
    return obs

def obs_edg_dgen(adj, q1, q2, q3):
    obsD = deg_distr(adj)
    obsE = obs_edg(adj, q1, q2, q3)
    obs = torch.cat((obsD[:q1], obsE))
    return obs

def avg_degreetype(adj, bustypes):
    dgen = 0
    dload = 0
    dint = 0
    count_gen = 0
    count_load = 0
    count_int = 0

    # Create a mask for each bus type
    for i in range(len(adj)):
        degree_sum = torch.sum(adj[i]).item()
        if bustypes[i] == 1:
            dgen += degree_sum
            count_gen += 1
        elif bustypes[i] == 2:
            dload += degree_sum
            count_load += 1
        else:
            dint += degree_sum
            count_int += 1

    # Avoid division by zero
    dgen /= count_gen if count_gen > 0 else 1
    dload /= count_load if count_load > 0 else 1
    dint /= count_int if count_int > 0 else 1
    
    return dgen, dload, dint, count_gen, count_load, count_int



def fast_obsDD(obs, move, i, j):
    dmove = 2 * move - 1
    obs[i] += dmove
    obs[j] += dmove
    return obs


 
def fast_obs_er_tri_2tri(past_obs, mtx, move, i, j):
    newobs = past_obs.clone()  # Use clone() instead of deepcopy
    t, kcount = change_triang3(mtx, i, j)
    if move == 1:
        newobs[0] += 1
        newobs[1] += t
        k = newobs[2] + kcount
        newobs[2] = k.clone()  # Use clone()
    else:
        newobs[0] -= 1
        newobs[1] -= t
        k = newobs[2] - kcount
        newobs[2] = k.clone()  # Use clone()
    return newobs


 
def fast_obs_edg(past_obs, mtx, ordlist, move, i, j):
    newobs = past_obs.clone()  # Use clone()
    addtype = linktype(ordlist[i], ordlist[j])
    newobs[addtype] += 1 if move == 1 else -1
    return newobs


 
def fast_obs_edg_tri_2tri(past_obs, mtx, ordlist, move, i, j):
    newobs = past_obs.clone()  # Use clone()
    t, kcount = change_triang3(mtx, i, j)
    if move == 1:
        addtype = linktype(ordlist[i], ordlist[j])
        newobs[addtype] += 1
        newobs[6] += t
        k = newobs[7] + kcount
        newobs[7] = k.clone()  # Use clone()
    else:
        addtype = linktype(ordlist[i], ordlist[j])
        newobs[addtype] -= 1
        newobs[6] -= t
        k = newobs[7] - kcount
        newobs[7] = k.clone()  # Use clone()
    return newobs


 
def alt_fast_obs_edg_tri_2tri(past_obs, mtx, ordlist, move, i, j):
    newobs = past_obs.clone()  # Use clone()
    t, kcount = change_triang3(mtx, i, j)
    k = 3 * t - kcount
    addtype = linktype(ordlist[i], ordlist[j])
    newobs[addtype] += 1 if move == 1 else -1
    newobs[6] += k if move == 1 else -k
    return newobs


 
def fast_obs_ddeg(past_obs, mtx, ordlist, move, i, j):
    newobs = deg_distr(mtx)
    return newobs


 
def fast_obs_edg_ddeg(past_obs, mtx, ordlist, move, i, j):
    newobsE = fast_obs_edg(past_obs[-6:], mtx, ordlist, move, i, j)
    newobsD = deg_distr(mtx)
    newobs = torch.cat((newobsD, newobsE))
    return newobs


 
def fast_obs_edg_dgen(past_obs, mtx, ordlist, move, i, j):
    newobsE = fast_obs_edg(past_obs[-6:], mtx, ordlist, move, i, j)
    newobsD = deg_distr(mtx)[:torch.bincount(ordlist)[1].item()]
    newobs = torch.cat((newobsD, newobsE))
    return newobs


def comp_obs_and_betas(modtype, ordmat, ordlist, countlist, maxiter=3000, startguess=torch.tensor([1.0, -0.2])):
    q1, q2, q3 = countlist[0], countlist[1], countlist[2]
    if modtype == '_edg_tri_2tri':
        obs = obs_edg_tri_2tri(ordmat, q1, q2, q3)
        kbetas = bc.greedsearch_param2(obs[:6], q1, q2, q3, maxiter=maxiter)
        betas = torch.cat((kbetas * 1.1, startguess))
        return obs, betas
    if modtype == '_edg_ddeg':
        kedgobs_T = obs_edg(ordmat, q1, q2, q3)
        realvals = deg_distr(ordmat)
        obs = torch.cat((realvals, kedgobs_T))
        thetasD, thetasE, dlist, elist = bc.greedsearch_paramDD(realvals, kedgobs_T, ordmat, ordlist, countlist, maxiter=300000)
        betas = torch.cat((thetasD, thetasE))
        return obs, betas
    if modtype == '_edg':
        obs = obs_edg(ordmat, q1, q2, q3)
        betas = bc.greedsearch_param2(obs, q1, q2, q3, maxiter=maxiter)
        return obs, betas
    if modtype == '_edg_dgen':
        kedgobs_T = obs_edg(ordmat, q1, q2, q3)
        realvals = deg_distr(ordmat)
        thetasD, thetasE, dlist, elist = bc.greedsearch_paramDD_gen(realvals, kedgobs_T, ordmat, ordlist, countlist, maxiter=300000)
        betas = torch.cat((thetasD[:q1], thetasE))
        obs = torch.cat((realvals[:q1], kedgobs_T))
        return obs, betas
