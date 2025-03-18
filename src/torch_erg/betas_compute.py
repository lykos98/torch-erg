import torch
from tqdm import tqdm
import math

def byn_coef(N):
    if N >= 2:
        return math.factorial(torch.tensor(N)) // (math.factorial(torch.tensor(N - 2)) * 2)
    else:
        return torch.tensor(0)

  
def sigmoid(th):
    return torch.exp(th) / (1 + torch.exp(th))

  
def linktype(a, b):
    return (a * b - 1) if (a + b < 5) else (a + b - 1)

def freederivative2(theta, N_der):
    return N_der * sigmoid(theta)

  
def tweak_params(exval, theta, N_der):
    theta += -0.025 if exval - freederivative2(theta, N_der) < 0 else 0.025
    return theta

def check_condition(realvals, thetas, quants):
    # Ensure all tensors are on the same device and dtype
    device = realvals.device
    thetas = thetas.to(device=device, dtype=torch.float64)
    quants = quants.to(device=device, dtype=torch.float64)
    
    # Compute the property
    prop = freederivative2(thetas, quants)
    
    # Compare with realvals within tolerance
    return torch.allclose(prop, realvals, atol=0.2)

  
def greedsearch_param2(realvals, NG, NL, NI, maxiter=30000, startguess=torch.zeros(6)):
    NGG = byn_coef(NG)
    NLL = byn_coef(NL)
    NII = byn_coef(NI)
    NGL = NG * NL
    NGI = NG * NI
    NLI = NL * NI
    quants = torch.tensor([NGG, NGL, NGI, NLL, NLI, NII])
    thetas = startguess.clone()
    condition = False

    for i in tqdm(range(maxiter)):
        for j in range(len(realvals)):
            thetas[j] = tweak_params(realvals[j], thetas[j], quants[j])
            condition = check_condition(realvals, thetas, quants)
            if condition:
                break

    if i >= (maxiter - 2):
        print('no convergence')
    
    for k in range(len(realvals)):
        print(freederivative2(thetas[k], quants[k]))
    
    return thetas

  
def greedsearch_paramDD(realvalsD, realvalsE, ordmat, ordlist, countlist, maxiter=300000):
    thetasD = torch.zeros(len(realvalsD))
    thetasE = torch.zeros(len(realvalsE))
    dlist = []
    elist = []

    for l in tqdm(range(maxiter)):
        for k in range(len(realvalsD)):
            thetasD = tweak_paramsD(realvalsD[k], k, ordlist, thetasE, thetasD)

        for z in range(len(realvalsE)):
            thetasE = tweak_paramsE(realvalsE[z], z, countlist, thetasE, thetasD)

    for k in range(len(realvalsD)):
        dlist.append(Dderivative(k, ordlist, thetasE, thetasD))
    for z in range(len(realvalsE)):
        elist.append(Ederivative(z, countlist, thetasE, thetasD))
    
    return thetasD, thetasE, dlist, elist

  
def greedsearch_paramDD_gen(realvalsD, realvalsE, ordmat, ordlist, countlist, maxiter=300000):
    thetasD = torch.zeros(len(realvalsD))
    thetasE = torch.zeros(len(realvalsE))
    dlist = []
    elist = []

    for l in tqdm(range(maxiter)):
        for k in range(countlist[0]):
            thetasD = tweak_paramsD(realvalsD[k], k, ordlist, thetasE, thetasD)

        for z in range(len(realvalsE)):
            thetasE = tweak_paramsE(realvalsE[z], z, countlist, thetasE, thetasD)

    for k in range(countlist[0]):
        dlist.append(Dderivative(k, ordlist, thetasE, thetasD))
    for z in range(len(realvalsE)):
        elist.append(Ederivative(z, countlist, thetasE, thetasD))
    
    return thetasD[:countlist[0]], thetasE, dlist, elist

def greedsearch_paramDD2(realvalsD, ordlist, countlist, maxiter=300000):
    thetasD = torch.zeros(len(realvalsD))
    dlist = []

    for l in tqdm(range(maxiter)):
        for k in range(len(realvalsD)):
            thetasD = tweak_paramsD2(realvalsD[k], k, thetasD)

    for k in range(len(realvalsD)):
        dlist.append(Dderivative2(k, thetasD))
    
    return thetasD, dlist

  
def reversetype(lt, countlist):
    if lt == 0:
        return (0, countlist[0]), (0, countlist[0])
    if lt == 1:
        return (0, countlist[0]), (countlist[0], countlist[0] + countlist[1])
    if lt == 2:
        return (0, countlist[0]), (countlist[0] + countlist[1], countlist[0] + countlist[1] + countlist[2])
    if lt == 3:
        return (countlist[0], countlist[0] + countlist[1]), (countlist[0], countlist[0] + countlist[1])
    if lt == 4:
        return (countlist[0], countlist[0] + countlist[1]), (countlist[0] + countlist[1], countlist[0] + countlist[1] + countlist[2])
    if lt == 5:
        return (countlist[0] + countlist[1], countlist[0] + countlist[1] + countlist[2]), (countlist[0] + countlist[1], countlist[0] + countlist[1] + countlist[2])
    return (0, 0), (0, 0)

  
def Dderivative(idx, ordlist, thetE, thetD):
    val = torch.sum(sigmoid(thetD[idx] + thetD + thetE[linktype(ordlist[idx], ordlist)])) - sigmoid(thetD[idx] + thetD[idx] + thetE[linktype(ordlist[idx], ordlist[idx])])
    return val

  
def Ederivative(bt, countlist, thetE, thetD):
    val = 0
    l1, l2 = reversetype(bt, countlist)
    
    if l1 != l2:
        for n in range(l1[0], l1[1]):
            for m in range(l2[0], l2[1]):
                val += sigmoid(thetD[n] + thetD[m] + thetE[bt])
    else:
        for n in range(l1[0], l1[1]):
            for m in range(n, l2[1]):
                val += sigmoid(thetD[n] + thetD[m] + thetE[bt])
    
    return val

  
def Dderivative2(idx, thetD):
    val = torch.sum(sigmoid(thetD[idx] + thetD)) - sigmoid(thetD[idx] + thetD[idx])
    return val

def tweak_paramsD(dval, idx, ordlist, thetE, thetD):
    if dval - Dderivative(idx, ordlist, thetE, thetD) < 0:
        thetD[idx] -= 0.025
    else:
        thetD[idx] += 0.025
    return thetD

def tweak_paramsE(Eval, bt, countlist, thetE, thetD):
    if Eval - Ederivative(bt, countlist, thetE, thetD) < 0:
        thetE[bt] -= 0.025
    else:
        thetE[bt] += 0.025
    return thetE

def tweak_paramsD2(dval, idx, thetD):
    if dval - Dderivative2(idx, thetD) < 0:
        thetD[idx] -= 0.025
    else:
        thetD[idx] += 0.025
    return thetD
