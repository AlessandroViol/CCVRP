import numpy as np
import pandas as pd
import gurobipy as gb
import matplotlib.pyplot as plt
import random
import time
from itertools import combinations


# Method that creates and returns a ccvpr model and its variables. Parameters of the problem instance are provided as arguments.
def CCVPR_model(
    sizeK,
    sizeI,
    sizeP,
    P,
    K,
    I,
    Tmax,
    Qmax,
    tu,
    cu,
    delta,
    vi_indexes,             # List of the indexes of the valid inequalities to add to the model.
    Qmax_k,
    selfConstr = False,     # Set to True to use the custom valid inequality X[i, i, k, p] = 0.
    TConstr = False,        # Set to True to use the custom valid inequality T[i, p] = 0.
    LConstr = False         # Set to True to use the custom valid inequality L[i, p] = 0.
):
    # Create gurobi model.
    print("-------------------------\n")
    ccvpr = gb.Model()
    ccvpr.modelSense = gb.GRB.MAXIMIZE
    print("-------------------------\n")

    # Build node matrix from K and I.
    N = pd.concat([K.drop(labels = ["V", "|A|", "Alpha", "R"], axis = 1),
                   I.drop(labels = ["s", "q", "service_in_periods", "Original_Carrier", "PI"], axis = 1)],
                  ignore_index = True)

    # Shorthand for the number of nodes in the matrix.
    sizeN = len(N.index)
    #print("[{}, {}] nodes, nodes matrix N:".format(sizeN, sizeN))
    #print(N)

    # The problem istances used provide only the coordinates of the nodes.
    # We assume costs c_ij and time t_ij directly depend on the distance between nodes.
    c = np.zeros([sizeN, sizeN])
    for i in range(sizeN):
        for j in range(sizeN):
            c[i, j] = np.linalg.norm(np.array([N.xCoord[i], N.yCoord[i]]) - 
                                     np.array([N.xCoord[j], N.yCoord[j]]))
    t = tu * c
    c = cu * c
    #print(c)


    # Varible declaration.

    # Binary variable taking value 1 if customer i is assigned to carrier k and value 0 otherwise.
    Y = ccvpr.addVars([(i, k) for i in range(sizeI) for k in range(sizeK)],
                        name = "Y",
                        vtype = gb.GRB.BINARY)

    # Binary variable taking value 1 if node j is visited immediately after node i by carrier k in period p and value 0 otherwise.
    X = ccvpr.addVars([(i, j, k, p) for i in range(sizeN) for j in range(sizeN) for k in range(sizeK) for p in range(sizeP)],
                      name = "X",
                      vtype = gb.GRB.BINARY)

    # Non-negative variable representing visit time of customer i on period p.
    # We use sizeN because we need to have visit times for the depots.
    T = ccvpr.addVars([(i, p) for i in range(sizeN) for p in range(sizeP)],
                      lb = 0.0,
                      name = "T",
                      vtype = gb.GRB.CONTINUOUS)

    # Non-negative variable representing cumulative load at node i in period p.
    L = ccvpr.addVars([(i, p) for i in range(sizeN) for p in range(sizeP)],
                      lb = 0.0,
                      name = "L",
                      vtype = gb.GRB.CONTINUOUS) #1879.7939178105573

    # Integer variable representing the minimum number of vehicles needed to fulfill the demand assigned to carrier k in period p. 
    Vmin = ccvpr.addVars([(k, p) for k in range(sizeK) for p in range(sizeP)],
                         lb = 0.0,
                         name = "Vmin",
                         vtype = gb.GRB.INTEGER)

    
    # Set objective function.

    # (expr. 1): maximizes the total profit, meaning the sum of collected revenues reduced by total travel costs
    ccvpr.setObjective(gb.quicksum(I.PI[i] for i in range(sizeI)) - 
                       gb.quicksum(gb.quicksum(gb.quicksum(gb.quicksum(c[i, j] * X[i, j, k, p]
                                                                       for i in range(sizeN))
                                                           for j in range(sizeN)) 
                                               for k in range(sizeK))
                                   for p in range(sizeP)))


    # Constraints declaration.

    # name_of_the_constraint (expr. expression_number_in_the_paper): description
    # C1 (expr. 2): impose that each customer is assigned to one and only one carrier.
    for i in range(sizeI):
        ccvpr.addConstr(gb.quicksum(Y[i, k] for k in range(sizeK)) == 1,
                        name = "C1_" + N.ID[i])

    # C2 (expr. 3): ensures that a customer can be visited by a carrier only if it has been assigned to it.
    for j in range(sizeI):
        for k in range(sizeK):
            for p in range(sizeP):
                # Because j in I, we need to use X[i, j + sizeK,...] otherwise we'll have j in a subset of N.
                ccvpr.addConstr(gb.quicksum(X[i, j + sizeK, k, p] for i in range(sizeN)) <= Y[j, k],
                                name = "C2_" + N.ID[j + sizeK] + K.ID[k] + P[p])

    # C3 (expr. 4): flow balance constraints.
    for j in range(sizeI):
        for k in range(sizeK):
            for p in range(sizeP):
                ccvpr.addConstr(gb.quicksum(X[i, j + sizeK, k, p] for i in range(sizeN)) == 
                                gb.quicksum(X[j + sizeK, i, k, p] for i in range(sizeN)),
                                name = "C3_" + N.ID[j + sizeK] + K.ID[k] + P[p])

    # C4 (expr. 5): fix the maximum number of vehicles a carrier can use.
    for k in range(sizeK):
        for p in range(sizeP):
            ccvpr.addConstr(gb.quicksum(X[j + sizeK, k, k, p] for j in range(sizeI)) <= K.V[k],
                            name = "C4_" + K.ID[k] + P[p])

    # C5 (expr. 6): arrival time at a customer.
    for j in range(sizeI):
        for i in range(sizeN):
            for p in range(sizeP):
                if i >= sizeK:
                    # We need to put i-sizeK because of the offset between I and the portion of N that corresponds to I.
                    ccvpr.addConstr(T[j + sizeK, p] >=
                                    T[i, p] + t[i, j + sizeK] + I.s[i - sizeK] * I.service_in_periods[i - sizeK][p] -
                                    Tmax * (1 - gb.quicksum(X[i, j + sizeK, k, p] for k in range(sizeK))),
                                    name = "C5_" + N.ID[j + sizeK] + N.ID[i] + P[p])
                else:
                    # For the depots no s is given, so we assume it is 1, meaning a vehicle needs to spend a minimum amount of time
                    # in the depot to collect a revenue.
                    ccvpr.addConstr(T[j + sizeK, p] >= 
                                    T[i, p] + t[i, j + sizeK] + 1 -
                                    Tmax * (1 - gb.quicksum(X[i, j + sizeK, k, p] for k in range(sizeK))),
                                    name = "C5_" + N.ID[j + sizeK] + N.ID[i] + P[p])

    ## C6 (expr. 7): route duration cannot exceed maximum allowed value Tmax.
    for j in range(sizeI):
        for i in range(sizeK):
            for p in range(sizeP):
                ccvpr.addConstr(t[j + sizeK, i] * gb.quicksum(X[j + sizeK, i, k, p] for k in range(sizeK)) <=
                                Tmax - T[j + sizeK, p],
                                name = "C6_" + N.ID[j + sizeK] + N.ID[i] + P[p])

    # C7 (expr. 8): cumulative load at a customer.
    for j in range(sizeI):
        for i in range(sizeN):
            for p in range(sizeP):
                ccvpr.addConstr(L[j + sizeK, p] >= L[i, p] + I.q[j] * I.service_in_periods[j][p] - 
                                Qmax * (1 - gb.quicksum(X[i, j + sizeK, k, p] for k in range(sizeK))),
                                name = "C7_" + N.ID[j + sizeK] + N.ID[i] + P[p])

    # C8 (expr. 9): ensure that the maximum loading capacity of the vehicles, Qmax_k, is respected.
    for j in range(sizeI):
        for p in range(sizeP):
            ccvpr.addConstr(L[j + sizeK, p] <= Qmax,
                            name = "C8_" + N.ID[j + sizeK] + P[p])

    # C9 (expr. 10): only vehicles owned by a carrier can exit the depot of that carrier.
    for k in range(sizeK):
        for i in range(sizeK):
            if i != k:
                for j in range(sizeI):
                    for p in range(sizeP):
                        ccvpr.addConstr(X[i, j + sizeK, k, p] == 0,
                                        name = "C9_" + N.ID[j + sizeK] + N.ID[i] + N.ID[k] + P[p])

    # C10 (expr. 11): only vehicles owned by a carrier can enter the depot of that carrier.
    for k in range(sizeK):
        for i in range(sizeK):
            if i != k:
                for j in range(sizeI):
                    for p in range(sizeP):
                        ccvpr.addConstr(X[j + sizeK, i, k, p] == 0,
                                        name = "C10_" + N.ID[i] + N.ID[j + sizeK] + N.ID[k] + P[p])

    # C11 (expr. 12): ensures that a customer that requires some good in a given period is served in that period.
    for j in range(sizeI):
        for p in range(sizeP):
            ccvpr.addConstr(gb.quicksum(gb.quicksum(X[i, j + sizeK, k, p] 
                                                    for k in range(sizeK)) 
                                        for i in range(sizeN)) >= I.q[j] * I.service_in_periods[j][p] / Qmax, 
                            name = "C11_" + N.ID[j + sizeK] + P[p])

    # C12 (expr. 13): operational constraint, fix the earliest starting time from the depot.
    for i in range(sizeK):
        for p in range(sizeP):
            ccvpr.addConstr(T[i, p] == 0, name = "C12_" + N.ID[i] + P[p])

    # C13 (expr. 14): operational constraint, fix the cumulative load at the depot to 0.
    for i in range(sizeK):
        for p in range(sizeP):
            ccvpr.addConstr(L[i, p] == 0, name = "C13_" + N.ID[i] + P[p])

    # C14 (expr. 18): paired with C15 ensures arrival times consistency over periods. Linearization of expr. 15.
    for j in range(sizeI):
        for p1 in range(sizeP):
            # I.q[j] * I.service_in_periods[j][p1] > 0 if and only if I.service_in_periods[j][p1] > 0
            if I.service_in_periods[j][p1] > 0:
                for p2 in range(sizeP):
                    if I.service_in_periods[j][p2] > 0 and p2 != p1:
                        ccvpr.addConstr(T[j + sizeK, p1] - T[j + sizeK, p2] <= delta,
                                        name = "C14_" + N.ID[j + sizeK] + P[p1] + P[p2])

    # C15 (expr. 19): paired with C14 ensures arrival times consistency over periods. Linearization of expr. 15.
    # This constraint is redundant with the one above
    #for j in range(sizeI):
    #    for p1 in range(sizeP):
    #        # I.q[j] * I.service_in_periods[j][p1] > 0 if and only if I.service_in_periods[j][p1] > 0
    #        if I.service_in_periods[j][p1] > 0:
    #            for p2 in range(sizeP):
    #                if I.service_in_periods[j][p2] > 0 and p2 != p1:
    #                    ccvpr.addConstr(T[j + sizeK, p2] - T[j + sizeK, p1] <= delta,
    #                                    name = "C15_" + N.ID[j + sizeK] + P[p2] + P[p1])

    # C16 (expr. 16): ensures that each carrier’s profit (=revenue - travel costs) must be equal or higher than the profit obtainable
    # without taking part in the coalition.
    for k in range(sizeK):
        ccvpr.addConstr(gb.quicksum(I.PI[j] * Y[j, k] for j in range(sizeI)) - 
                        gb.quicksum(gb.quicksum(gb.quicksum(c[i, j + sizeK] * X[i, j + sizeK, k, p]
                                                            for j in range(sizeI))
                                                for i in range(sizeN))
                                    for p in range(sizeP)) >= K.R[k],
                        name = "C16_" + K.ID[k])

    # C17 (expr. 17): maintain workload balance, the number of customers assigned to a given carrier cannot be lower than
    # a minimum value imposed by the carrier.
    for k in range(sizeK):
        ccvpr.addConstr(gb.quicksum(Y[j, k] for j in range(sizeI)) >= K["|A|"][k] - K.Alpha[k],
                        name = "C17_" + K.ID[k])

    # C18 (expr. 20): the minimum number of vehicles required to fulfill the demand of a carrier k in period p, Vmin[k, p] is bounded.
    for k in range(sizeK):
        for p in range(sizeP):
            ccvpr.addConstr(Vmin[k, p] >= 
                            gb.quicksum(I.q[i] * I.service_in_periods[i][p] * Y[i, k] for i in range(sizeI)) / Qmax_k[k],
                            name = "C18_" + K.ID[k] + P[p])
    

    # Valid inequalities declaration.

    # Cvi1 (expr. 21): implies that the sum of the revenues associated with the customers assigned to a specific carrier k
    # must be greater than the best profit R[k] obtainable by this carrier without collaboration.
    if 1 in vi_indexes:
        for k in range(sizeK):
            ccvpr.addConstr(gb.quicksum(I.PI[i] * Y[i, k] for i in range(sizeI)) >= K.R[k],
                            name = "Cvi1_" + K.ID[k])

        print("Valid inequality 1 (expr. 21) added")

    # Cvi2 (expr. 22): prevent assignments to a given carrier, where the total demand for a period exceeds the maximum
    # demand manageable by the carrier in a single period.
    if 2 in vi_indexes:
        for k in range(sizeK):
            for p in range(sizeP):
                ccvpr.addConstr(gb.quicksum(I.q[i] * I.service_in_periods[i][p] * Y[i, k] for i in range(sizeI)) <=
                                K.V[k] * Qmax_k[k],
                                name = "Cvi2_" + K.ID[k] + P[p])
                
        print("Valid inequality 2 (expr. 22) added")

    # Cvi3 (expr. 23): upper bound on the maximum number of customers that can be assigned to a given carrier.
    if 3 in vi_indexes:
        for k in range(sizeK):
            capacity = K["|A|"][k] + np.sum(K.Alpha) - K.Alpha[k]
            if sizeI <= capacity:
                max_customers = sizeI
            else:
                max_customers = capacity
        
            ccvpr.addConstr(gb.quicksum(Y[i, k] for i in range(sizeI)) <= max_customers, name = "Cvi3_" + K.ID[k])

        print("Valid inequality 3 (expr. 23) added")

    # Cvi4 (expr. 24): tight lower and upper bounds on the number of X[i, j, k, p] variables having simultaneously a value of 1 for 
    # each carrier k and each period p, given the number of customers that have been assigned to k and that have to be served in period p. 
    if 4 in vi_indexes:
        for k in range(sizeK):
            for p in range(sizeP):
                ccvpr.addConstr(gb.quicksum(Y[i, k] * I.service_in_periods[i][p] for i in range(sizeI)) + Vmin[k, p] <= 
                                gb.quicksum(gb.quicksum(X[i, j, k, p] 
                                                        for j in range(sizeN))
                                            for i in range(sizeN)),
                                name = "Cvi4A_" + K.ID[k] + P[p])

                ccvpr.addConstr(gb.quicksum(gb.quicksum(X[i, j, k, p] 
                                                        for j in range(sizeN))
                                            for i in range(sizeN)) <= 
                                gb.quicksum(Y[i, k] * I.service_in_periods[i][p] for i in range(sizeI)) + K.V[k],
                                name = "Cvi4B_" + K.ID[k] + P[p])

        print("Valid inequality 4 (expr. 24) added")


    # Custom constraints: we believe these basic constraints are implied in the model discussed in order for it to work or allow better performances.

    # Ensure we can't go from node i to node i.
    if selfConstr:
        for p in range(sizeP):
            for k in range(sizeK):
                for i in range(sizeN):
                    ccvpr.addConstr(X[i, i, k, p] == 0, name = "C_self_" + N.ID[i] + N.ID[i] + K.ID[k] + P[p])

    # Set the variable T[i, p] to 0 if customer i doesn't need to be served in period p.
    if TConstr:
        for p in range(sizeP):
            for i in range(sizeI):
                if I.service_in_periods[i][p] <= 0:
                    ccvpr.addConstr(T[i + sizeK, p] == 0)

    # Set the variable L[i, p] to 0 if customer i doesn't need to be served in period p.
    if LConstr:
        for p in range(sizeP):
            for i in range(sizeI):
                if I.service_in_periods[i][p] <= 0:
                    ccvpr.addConstr(L[i + sizeK, p] == 0)

    # Writes in a file all the variables, constraints and objective functions defined in the model for manual inspection.
    ccvpr.write("Istances/ConstraintControl.lp")
    
    return N, sizeN, t, c, Y, X, T, L, Vmin, ccvpr


# Load the values of the specified problem instance into some variables and pandas dataframe. The input file should be a .txt file with the 
# characteristics descibed in the read_me.txt file.
def import_problem_istance(path):
    print("Loading problem istance " + path)

    # Read the first row of the instance file, which contains the number of carriers, customers and periods.
    values = pd.read_csv(path, sep = "\t", header = None, nrows = 1, usecols = range(3))
    values = values.to_numpy()
    sizeK = values[0, 0]
    sizeI = values[0, 1]
    sizeP = values[0, 2]

    # Build a list of periods for utility purpuses. 
    P = []
    for p in range(sizeP):
        P = np.append(P, "p" + str(p + 1))
    print("Size K = {};\t Size I = {};\t Size P = {}, P: {}".format(sizeK, sizeI, sizeP, P))

    # Read the second row of the instance file, which contains the maximum route duration and the maximum cumulative load.
    values = pd.read_csv(path, sep = "\t", header = None, skiprows = 1, nrows = 1, usecols = range(2))
    values = values.to_numpy()
    Tmax = values[0, 0]
    Qmax = values[0, 1]
    print("Tmax = {};\t Qmax = {}".format(Tmax, Qmax))

    # Read the next sizeK rows of the instance file, which contain the informations about the carriers and their depots.
    K = pd.read_csv(path, sep = "\t", header = None, names = ["ID", "xCoord", "yCoord", "V", "|A|", "Alpha"],
                    skiprows = 2, nrows = sizeK, usecols  =range(6))
    
    # Create and adds a list of IDs for the carriers to K for ease of use.
    ids = []
    for k in range(sizeK):
        ids = np.append(ids, "k" + str(k + 1))
    K["ID"] = ids

    # Read the last row of the instance file, which contains the values of R for each carrier. These values are then added to K.
    values = pd.read_csv(path, sep = " ", header = None, skiprows = 2 + sizeK + sizeI, nrows = 1, usecols = range(sizeK))
    R = values.to_numpy()
    K["R"] = np.transpose(R)
    print("Carriers:")
    print(K)

    # Read the sizeI rows of the instance file which contain the informations about the customers.
    I = pd.read_csv(path, sep = "\t", header = None, skiprows = 2 + sizeK, nrows = sizeI, usecols = range(8),
                    names = ["ID", "xCoord", "yCoord", "s", "q", "service_in_periods", "Original_Carrier", "PI"])

    # Create and adds a list of IDs for the customers to I for ease of use. Also, the string that represents if service is needed for
    # a customer in a period is converted to a list of integers.
    ids = []
    srp = []
    for i in range(sizeI):
        ids = np.append(ids, "i" + str(i + 1))
        srp = np.append(srp, [*I["service_in_periods"].to_numpy()[i].replace(" ", "")])
    srp = np.reshape(srp, [sizeI, np.size([*I["service_in_periods"].to_numpy()[0].replace(" ", "")])])
    I["ID"] = ids
    I["service_in_periods"] = srp.astype(int).tolist()
    print("Customers:")
    print(I)

    return sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax


# Prints out an estimate of the total route duration for every route of every carrier in every period for the instance pr01_20. 
# A candidate value for the time per space unit is used. Allows us to check if a route exceeds the maximum duration Tmax.
def estimate_time_per_unit(K, I, Tmax, candidate):
    # Build node matrix from K and I.
    N = pd.concat([K.drop(labels = ["V", "|A|", "Alpha", "R"], axis = 1),
                   I.drop(labels = ["s", "q", "service_in_periods", "Original_Carrier", "PI"], axis = 1)],
                  ignore_index=True)

    # Shorthand for the number of nodes in the matrix.
    sizeN = len(N.index)
    
    c = np.zeros([sizeN, sizeN])
    for i in range(sizeN):
        for j in range(sizeN):
            c[i, j] = np.linalg.norm(np.array([N.xCoord[i], N.yCoord[i]]) - 
                                     np.array([N.xCoord[j], N.yCoord[j]]))
            
    # Try to get an estimate of time cost per unit by checking the route durations for the solution of pr01_20
    time_p1_k1 = c[0, 10] + c[10, 12] + c[12, 0]
    service_time = I.s[6] + I.s[8]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p1_k1, service_time, candidate * time_p1_k1 + service_time, Tmax))
    
    time_p1_k2 = c[1, 9] + c[9, 14] + c[14, 1]
    service_time = I.s[5] + I.s[10]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p1_k2, service_time, candidate * time_p1_k2 + service_time, Tmax))
    
    time_p1_k3 = c[2, 7] + c[7, 4] + c[4, 11] + c[11, 16] + c[16, 20] + c[20, 2]
    service_time = I.s[3] + I.s[0] + I.s[7] + I.s[12] + I.s[16] 
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p1_k3, service_time, candidate * time_p1_k3 + service_time, Tmax))
    
    time_p1_k4 = c[3, 5] + c[5, 15] + c[15, 3]
    service_time = I.s[1] + I.s[11]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p1_k4, service_time, candidate * time_p1_k4 + service_time, Tmax))
    
    time_p2_k1 = c[0, 13] + c[13, 10] + c[10, 0]
    service_time = I.s[6] + I.s[9]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p2_k1, service_time, candidate * time_p2_k1 + service_time, Tmax))
    
    time_p2_k2 = c[1, 9] + c[9, 6] + c[6, 1] 
    service_time = I.s[10] + I.s[5] 
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p2_k2, service_time, candidate * time_p2_k2 + service_time, Tmax))
    
    time_p2_k4 = c[3, 18] + c[18, 21] + c[21, 8] + c[8, 19] + c[19, 17] + c[17, 3]
    service_time = I.s[14] + I.s[17] + I.s[4] + I.s[15] + I.s[13]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p2_k4, service_time, candidate * time_p2_k4 + service_time, Tmax))
    
    time_p3_k1 = c[0, 13] + c[13, 22] + c[22, 0]
    service_time = I.s[9] + I.s[18]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p3_k1, service_time, candidate * time_p3_k1 + service_time, Tmax))
    
    time_p3_k2 = c[1, 14] + c[14, 9] + c[9, 6] + c[6, 1]
    service_time = I.s[10] + I.s[5] + I.s[2]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p3_k2, service_time, candidate * time_p3_k2 + service_time, Tmax))
    
    time_p3_k3 = c[2, 7] + c[7, 16] + c[16, 11] + c[11, 2]
    service_time = I.s[3] + I.s[12] + I.s[7]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p3_k3, service_time, candidate * time_p3_k3 + service_time, Tmax))
    
    time_p3_k4 = c[3, 15] + c[15, 3] 
    service_time = I.s[11]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p3_k4, service_time, candidate * time_p3_k4 + service_time, Tmax))
    
    time_p4_k1 = c[0, 10] + c[10, 12] + c[12, 0]
    service_time = I.s[6] + I.s[8]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p4_k1, service_time, candidate * time_p4_k1 + service_time, Tmax))
    
    time_p4_k3 = c[2, 4] + c[4, 23] + c[23, 11] + c[11, 2]
    service_time = I.s[0] + I.s[19] + I.s[7]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p4_k3, service_time, candidate * time_p4_k3 + service_time, Tmax))
    
    time_p4_k4 = c[3, 5] + c[5, 17] + c[17, 19] + c[19, 8] + c[8, 18] + c[18, 3]
    service_time = I.s[1] + I.s[13] + I.s[15] + I.s[4] + I.s[14]
    print("({}) * t + {} \t=\t {} <= {}".format(candidate * time_p4_k4, service_time, candidate * time_p4_k4 + service_time, Tmax))


# Computes the value of the cost per space unit by using the information provided in the paper about the instance pr01_20.
def estimate_cost_per_unit(K, I):
    # Build node matrix from K and I.
    N = pd.concat([K.drop(labels = ["V", "|A|", "Alpha", "R"], axis = 1),
                   I.drop(labels = ["s", "q", "service_in_periods", "Original_Carrier", "PI"], axis = 1)],
                  ignore_index=True)

    # Shorthand for the number of nodes in the matrix.
    sizeN = len(N.index)
    
    c = np.zeros([sizeN, sizeN])
    for i in range(sizeN):
        for j in range(sizeN):
            c[i, j] = np.linalg.norm(np.array([N.xCoord[i], N.yCoord[i]]) - 
                                     np.array([N.xCoord[j], N.yCoord[j]]))
            
    # The cost per unit and time per unit values aren't provided. An estimate of the cost can be obtained from 
    # the bks of pr01_20 provided in the paper.
    cost_p1_k1 = c[0, 10] + c[10, 12] + c[12, 0]
    cost_p1_k2 = c[1, 9] + c[9, 14] + c[14, 1]
    cost_p1_k3 = c[2, 7] + c[7, 4] + c[4, 11] + c[11, 16] + c[16, 20] + c[20, 2]
    cost_p1_k4 = c[3, 5] + c[5, 15] + c[15, 3]
    cost_p1 = cost_p1_k1 + cost_p1_k2 + cost_p1_k3 + cost_p1_k4

    cost_p2_k1 = c[0, 13] + c[13, 10] + c[10, 0]
    cost_p2_k2 = c[1, 9] + c[9, 6] + c[6, 1]
    cost_p2_k3 = 0
    cost_p2_k4 = c[3, 18] + c[18, 21] + c[21, 8] + c[8, 19] + c[19, 17] + c[17, 3]
    cost_p2 = cost_p2_k1 + cost_p2_k2 + cost_p2_k3 + cost_p2_k4

    cost_p3_k1 = c[0, 13] + c[13, 22] + c[22, 0]
    cost_p3_k2 = c[1, 14] + c[14, 9] + c[9, 6] + c[6, 1]
    cost_p3_k3 = c[2, 7] + c[7, 16] + c[16, 11] + c[11, 2]
    cost_p3_k4 = c[3, 15] + c[15, 3]
    cost_p3 = cost_p3_k1 + cost_p3_k2 + cost_p3_k3 + cost_p3_k4

    cost_p4_k1 = c[0, 10] + c[10, 12] + c[12, 0]
    cost_p4_k2 = 0
    cost_p4_k3 = c[2, 4] + c[4, 23] + c[23, 11] + c[11, 2]
    cost_p4_k4 = c[3, 5] + c[5, 17] + c[17, 19] + c[19, 8] + c[8, 18] + c[18, 3]
    cost_p4 = cost_p4_k1 + cost_p4_k2 + cost_p4_k3 + cost_p4_k4

    total = cost_p1 + cost_p2 + cost_p3 + cost_p4
    gain = np.sum(I.PI)
    bks = 1743.43
    print("Total distance: {};\t Maximum gain: {};\t Expected revenue: {};\t Obtained cost per unit: {}".format(total, gain, bks, (gain - bks) / total))


# Use matheuristic MH to solve the provided model. Solved model is returned.
def CCVPR_matheuristic(model, Y, TLinit, TL, sizeK, sizeI, K, I):
    # Set TimeLimit parameter to TLinit and find a solution.
    model.setParam("TimeLimit", TLinit)
    model.optimize()

    # If no feasible solution has been found, we use a solution with each carrier serving only its original customers.
    if model.SolCount == 0:
        print("MH: No solution found in TLinit " + str(TLinit) + "s")

        for i in range(sizeI):
            for k in range(sizeK):
                # Using constraints, we properly assign each customer to its carrier.
                if I.Original_Carrier[i] - 1 == k:
                    model.addConstr(Y[i, k] == 1, name = "CMH1_" + I.ID[i] + K.ID[k])
                else:
                    model.addConstr(Y[i, k] == 0, name = "CMH1_" + I.ID[i] + K.ID[k])

        # After obtaining the solution the constraints added above are removed.
        model.optimize()
        for i in range(sizeI):
            for k in range(sizeK):
                model.remove(model.getConstrByName("CMH1_" + I.ID[i] + K.ID[k]))

    print("MH: Initial solution found")
    print("MH: initial objective value = {}".format(model.getAttr("ObjVal")))

    # The current solution is saved as current best solution.
    current_best = model.getAttr("ObjVal")

    #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
    #plt.title("Initial MH solution")

    # Set TimeLimit to TL and iterate until we get through Nnoimp iterations without improvement
    model.setParam("TimeLimit", TL)
    niter = 0
    noimp = 0
    all_pairs = list(combinations(range(sizeK), 2))
    
    while noimp < len(all_pairs):
        # Select every pair of carriers.
        k1 = all_pairs[noimp][0]
        k2 = all_pairs[noimp][1]
        print("MH: Non fixed carriers: {} and {}".format(K.ID[k1], K.ID[k2]))

        # Fix the assignment (and so the routing) for all the other carriers using contraints.
        for k in range(sizeK):
            if k != k1 and k != k2:
                for i in range(sizeI):
                    model.addConstr(Y[i, k] == Y[i, k].x, 
                                    name = "CMH1_" + I.ID[i] + K.ID[k])

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Old solution of MH iteration {}".format(niter))

        # Find new solution for the partial model. Revert to old solution if the one found is worst, keep it as new best otherwise.
        model.optimize()
        print("MH: (niter = {}) Best so far: {}\tCurrent: {}".format(niter + 1, current_best, model.getAttr("ObjVal")))

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("New solution of MH iteration {}".format(niter))

        if current_best < model.getAttr("ObjVal"):
            print("------------------------------------------------------------------------------------------")
            print("new best!")
            print("------------------------------------------------------------------------------------------\n")

            #model.write("tempBestMH.sol")
            current_best = model.getAttr("ObjVal")
            noimp = 0
        else:
            print("------------------------------------------------------------------------------------------")
            print("Old is better!")
            print("------------------------------------------------------------------------------------------\n")
            
            noimp = noimp + 1

        print("noimp set to " + str(noimp))
        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Best solution of MH iteration {}".format(niter))
        #plt.show()
        
        # Remove the constraints added above, hence freeing the routing for alla carriers. 
        for k in range(sizeK):
            if k != k1 and k != k2:
                for i in range(sizeI):
                    model.remove(model.getConstrByName("CMH1_" + I.ID[i] + K.ID[k]))

        # Commit changes to the model
        model.setParam("TimeLimit", .01)
        model.optimize()
        model.setParam("TimeLimit", TL)

        niter = niter + 1
                
    return model


# Use matheuristic MH* to solve the provided model. If use_ILS_init is True, the solution in the model is used as initial solution. 
# Solved model is returned.
def CCVPR_matheuristic_star(model, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I, use_ILS_init = False):
    # If use_ILS_init == True the ILS provided initial solution is used.
    if use_ILS_init == False:
        print("MH*: No ILS init, searching a solution in TLinit " + str(TLinit) + "s")
        # Set TimeLimit parameter to TLinit and find a solution.
        model.setParam("TimeLimit", TLinit)
        model.optimize()

        # If no feasible solution has been found, we use a solution with each carrier serving only its original customers.
        if model.SolCount == 0:
            print("MH*: No solution found in TLinit " + str(TLinit) + "s")

            for i in range(sizeI):
                for k in range(sizeK):
                    # Using constraints, we properly assign each customer to its carrier.
                    if I.Original_Carrier[i] - 1 == k:
                        model.addConstr(Y[i, k] == 1, name = "CMH1_" + I.ID[i] + K.ID[k])
                    else:
                        model.addConstr(Y[i, k] == 0, name = "CMH1_" + I.ID[i] + K.ID[k])

            #model.setParam("TimeLimit", 20)

            # After obtaining the solution the constraints added above are removed.
            model.optimize()
            for i in range(sizeI):
                for k in range(sizeK):
                    model.remove(model.getConstrByName("CMH1_" + I.ID[i] + K.ID[k]))
            model.setParam("TimeLimit", .01)
            model.optimize()

            print("MH*: Initial solution found")

    else:
        print("MH*: Obtained initial solution from ILS")

    print("MH*: initial objective value = {}".format(model.getAttr("ObjVal")))

    # The current solution is saved as current best solution.
    current_best = model.getAttr("ObjVal")

    #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
    #plt.title("Initial MH solution")

    # Set TimeLimit to TL and iterate until we get through Nnoimp iterations without improvement
    model.setParam("TimeLimit", TL)
    noimp = 0
    niter = 0

    while noimp < Nnoimp:
        # Select two carriers randomly from K.
        [k1, k2] = random.sample(range(sizeK), 2)
        print("MH*: Non fixed carriers: {} and {}".format(K.ID[k1], K.ID[k2]))

        # Fix the assignment (and so the routing) for all the other carriers using contraints.
        for k in range(sizeK):
            if k != k1 and k != k2:
                for i in range(sizeI):
                    model.addConstr(Y[i, k] == Y[i, k].x, 
                                    name = "CMH1_" + I.ID[i] + K.ID[k])

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Old solution of MH iteration {}".format(niter))

        # Find new solution for the partial model. Revert to old solution if the one found is worst, keep it as new best otherwise.
        model.optimize()
        print("MH*: (noimp = {}, niter = {}) Best so far: {}\tCurrent: {}".format(noimp + 1, niter + 1, current_best, model.getAttr("ObjVal")))

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("New solution of MH iteration {}".format(niter))

        if current_best < model.getAttr("ObjVal"):
            print("------------------------------------------------------------------------------------------")
            print("new best!")
            print("------------------------------------------------------------------------------------------\n")
            
            current_best = model.getAttr("ObjVal")

            # Restet no improvement counter.
            noimp = 0
        else:
            print("------------------------------------------------------------------------------------------")
            print("Old is better!")
            print("------------------------------------------------------------------------------------------\n")

            # Increase no improvement counter.
            noimp = noimp + 1

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Best solution of MH iteration {}".format(niter))
        #plt.show()
        
        # Remove the constraints added above, hence freeing the routing for alla carriers. 
        for k in range(sizeK):
            if k != k1 and k != k2:
                for i in range(sizeI):
                    model.remove(model.getConstrByName("CMH1_" + I.ID[i] + K.ID[k]))

        # Commit changes to the model
        model.setParam("TimeLimit", .01)
        model.optimize()
        model.setParam("TimeLimit", TL)

        niter = niter + 1

    return model


def CCVPR_iterated_local_search(model, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert):
    print("ILS: Starting ILS, search initial feasible solution with MH\n")

    # Obtain an initial solution using the matheuristic and save it as current best.
    model = CCVPR_matheuristic_star(model, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I)
    current_best = model.getAttr("ObjVal")
    model.write("tempBestILS.sol")

    print("ILS: Initial solution for ILS found with MH")

    #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
    #plt.title("Initial solution of ILS")

    iter = 1
    while iter < Niter:
        print("ILS: Iteration {} of ILS".format(iter + 1))

        # Select a random subset of customers and impose that they change their carrier.
        nsol = 0
        nperm = 0

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Solution of ILS before perm {}".format(iter))

        while nsol == 0:
            print("ILS: ILS perm iteration {}".format(nperm + 1))

            nperm = nperm + 1
            ipert = random.sample(range(sizeI), Npert)

            counter = 0
            for k in range(sizeK):
                for i in ipert:
                    # To force the customer to change carrier, a constraint is used to ensure it can't be assigned to that carrier.
                    if Y[i, k].X == 1:
                        model.addConstr(Y[i, k] == 0, name = "CILS1_" + str(counter))
                        counter = counter + 1

            model.setParam("TimeLimit", TLpert)
            model.optimize()

            # Remove the constraints added above and commit changes to the model.
            for i in range(counter):
                model.remove(model.getConstrByName("CILS1_" + str(i)))
            model.setParam("TimeLimit", .01)
            model.optimize()

            # If for this permutation no feasible solution is found, another permutation is tried.
            nsol = model.SolCount
            if nsol == 0:
                model.read("tempBestILS.sol")
            model.setParam("TimeLimit", .01)
            model.optimize()

        print("ILS: Found a solution for the subset of clients in iteration {} of ILS".format(iter + 1))

        # Current best solution is kept.
        if current_best < model.getAttr("ObjVal"):
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("New best!")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

            model.write("tempBestILS.sol")
            current_best = model.getAttr("ObjVal")
        else:
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("Old is better!")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

            model.read("tempBestILS.sol")
            model.setParam("TimeLimit", .01)
            model.optimize()
        
        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Best ILS solution at iteration {} for starting MH".format(iter))

        print("ILS: Search a better solution with MH in iteration {} of ILS".format(iter + 1))
        
        # Use current best solution as initial solution and run matheuristic.
        model = CCVPR_matheuristic_star(model, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I, use_ILS_init = True)

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("MH solution at ILS iteration {}".format(iter))

        # Current best solution is kept.
        if current_best < model.getAttr("ObjVal"):
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("New best!")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

            model.write("tempBestILS.sol")
            current_best = model.getAttr("ObjVal")
        else:
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("Old is better!")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

            model.read("tempBestILS.sol")
            model.setParam("TimeLimit", .01)
            model.optimize()

        #CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
        #plt.title("Selected solution at ILS iteration {}".format(iter))
        #plt.show()

        # Increase iteration counter.
        iter = iter + 1

    return model


# Return a plot of the routes for each period and each carrier for the provided solved model. Original carriers and depots are marked as well.
def CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X):
    color = ["red", "green", "blue", "orange", "slategray", "magenta", "gold", "forestgreen"]
    fig = plt.figure()

    for p in range(sizeP):
        # Create appropriately subplots.
        plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), p + 1)
        for k in range(sizeK):
            nodes = []
            for i in range(sizeN):
                if i >= sizeK:
                    # Plot original carrier if customer needs to be served in that period.
                    if I.service_in_periods[i - sizeK][p] > 0:
                        plt.plot(N.xCoord[i], N.yCoord[i], "+", c = color[I.Original_Carrier[i - sizeK] - 1])
                
                # Build a list of pair of nodes visited in a route.
                for j in range(sizeN):
                    if X[i, j, k, p].Xn == 1:
                        nodes = np.append(nodes, [i, j])
                        nodes = np.reshape(nodes, [-1, 2])

            # Plot depots and their label.
            plt.text(N.xCoord[k] + 1, N.yCoord[k] + 1, " " + N.ID[k], c = color[k])
            plt.plot(N.xCoord[k], N.yCoord[k], 's', c = color[k])

            # Arrange the nodes in the list so that the node j is the same of node i of the next element of the list.
            if nodes != []:
                for n1 in range(np.size(nodes, 0) - 1):
                    for n2 in range(n1 + 2, np.size(nodes, 0)):
                        if nodes[n2, 0] == nodes[n1, 1]:
                            aux = 1 * nodes[n1 + 1, :]
                            nodes[n1 + 1, :] = nodes[n2, :]
                            nodes[n2, :] = aux

                # Plot the route.
                for n in range(np.size(nodes, 0)):
                    x = [N.xCoord[nodes[n, 0]], N.xCoord[nodes[n, 1]]]
                    y = [N.yCoord[nodes[n, 0]], N.yCoord[nodes[n, 1]]]

                    plt.plot(x, y, c = color[k])

                    if nodes[n, 0] >= sizeK:
                        plt.text(x[0] + 1, y[0] + 1, " " + N.ID[nodes[n, 0]], c = color[k])
                        plt.plot(x[0], y[0], 'o', c = color[k])

                    if nodes[n, 1] >= sizeK:
                        plt.text(x[1] + 1, y[1] + 1, " " + N.ID[nodes[n, 1]], c = color[k])
                        plt.plot(x[1], y[1], 'o', c = color[k])

    return fig


# Use a solver to solve all the small problem instances. Results are saved in a .txt file and solutions in a .sol file.
def solve_small_instances(solver, save_fig = False, vi_indexes = [1, 3, 4]):
    # Define some common parameters not present in the instance file.
    tu = 0.7
    cu = 0.1
    delta = 60
    TLinit = 10
    TL = 10
    TLpert = 10
    Npert = 3
    Niter = 5
    Nnoimp = 5

    # Create the list of paths of the instances.
    file = []
    for i in range(9):
        file = np.append(file, "pr0" + str(i + 1) + "_20")
    file = np.append(file, "pr10_20")

    # Prapare a dataframe to store the values.
    df = pd.DataFrame(columns = ["Solution Value", "Gap (%)", "Time (s)"], index = file)
    df.columns.name = "Instance"

    for i in range(10):
        # Import the i-th problem instance.
        sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax = import_problem_istance("Istances/20 customers/{}.txt".format(file[i]))
        
        Qmax_k = np.full(sizeK, Qmax)
        print("Qmax_k: {}".format(Qmax_k))

        # Obtain the model and variables.
        N, sizeN, t, c, Y, X, T, L, Vmin, ccvprModel = CCVPR_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = vi_indexes,
                                                                   Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)
        ccvprModel.setParam("LogToConsole", 0)

        # Select solver.
        if solver == "MIP":
            print("Optimizing " + file[i] + " with MIP")

            ccvprModel.setParam("TimeLimit", 10 * 60)
            ccvprModel.optimize()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(ccvprModel.ObjVal, 2), 
                                         "Gap (%)": round(ccvprModel.MIPGap * 100, 2), 
                                         "Time (s)": round(ccvprModel.Runtime, 2)})

            # Write the solution in a .sol file.
            if vi_indexes == []:
                ccvprModel.write("Results/Small Instances/MIP/" + file[i] + "_no_VI.sol")
            else:
                ccvprModel.write("Results/Small Instances/MIP/" + file[i] + ".sol")

            # Create a plot of the solution routing.
            if save_fig:
                CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
                plt.savefig("Results/Small Instances/MIP/" + file[i] + ".png")

            print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvprModel.ObjVal, 2), 
                                                                        round(ccvprModel.MIPGap * 100, 2), 
                                                                        round(ccvprModel.Runtime, 2)))

        elif solver == "MH":
            print("Optimizing " + file[i] + " with MH")

            # Import the BKS obtained from the MIP to compute the relative gap.
            BKS = pd.read_csv("Results/Small Instances/MIP/MIP.txt", sep=";")
            BKS = BKS["Solution Value"][i]

            t_in = time.time()
            ccvprModel = CCVPR_matheuristic(ccvprModel, Y, TLinit, TL, sizeK, sizeI, K, I)
            t_fin = time.time()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(ccvprModel.ObjVal, 2),
                                         "Gap (%)": round((ccvprModel.ObjVal - BKS) * 100 / BKS, 2), 
                                         "Time (s)": round(t_fin - t_in, 2)})

            # Write the solution in a .sol file.
            ccvprModel.write("Results/Small Instances/MH/" + file[i] + ".sol")

            print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvprModel.ObjVal, 2), 
                                                                        round((ccvprModel.ObjVal - BKS) * 100 / BKS, 2), 
                                                                        round(t_fin - t_in, 2)))

        elif solver == "ILS":
            # Import the BKS obtained from the MIP to compute the relative gap.
            BKS = pd.read_csv("Results/Small Instances/MIP/MIP.txt", sep=";")
            BKS = BKS["Solution Value"][i]

            result = []
            runtime = []
            # Do 10 iterations and use the mean results.
            for n in range(10):
                print("Optimizing " + file[i] + " with ILS: iteration " + str(n + 1) + "/5")

                t_in = time.time()
                ccvprModel = CCVPR_iterated_local_search(ccvprModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
                t_fin = time.time()

                # Write the solution in a .sol file.
                ccvprModel.write("Results/Small Instances/ILS/" + file[i] + "_it_" + str(n + 1) + ".sol")
                
                result = np.append(result, ccvprModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvprModel.ObjVal, 2), 
                                                                            round((ccvprModel.ObjVal - BKS) * 100 / BKS, 2), 
                                                                            round(t_fin - t_in, 2)))
                ccvprModel.reset()
            
            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(np.mean(result), 2),
                                         "Gap (%)": round((np.mean(result) - BKS) * 100 / BKS, 2), 
                                         "Time (s)": round(np.mean(runtime), 2)})

    # Save the dataframe of the results to a .txt file.
    df.to_csv("Results/Small Instances/" + solver + "/" + solver + ".txt", sep = ";", index_label = df.columns.name)
    print(df)


# Use a solver to solve all the large problem instances. Results are saved in a .txt file and solutions in a .sol file.
def solve_large_instances(solver):
    # Define some common parameters not present in the instance file.
    tu = 0.7
    cu = 0.1
    delta = 60
    TLinit = 10
    TL = 10
    TLpert = 10
    Npert = 3
    Niter = 5
    Nnoimp = 5

    # Create the list of paths of the instances.
    file = []
    for i in range(9):
        file = np.append(file, "pr0" + str(i + 1) + "_50")
    file = np.append(file, "pr10_50")

    # Prapare a dataframe to store the values.
    df = pd.DataFrame(columns = ["Solution Value", "Time (s)"], index = file)
    df.columns.name = "Instance"

    for i in range(10):
        # Import the i-th problem instance.
        sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax = import_problem_istance("Istances/50 customers/{}.txt".format(file[i]))
        
        Qmax_k = np.full(sizeK, Qmax)
        print("Qmax_k: {}".format(Qmax_k))

        # Obtain the model and variables.
        N, sizeN, t, c, Y, X, T, L, Vmin, ccvprModel = CCVPR_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = [1, 3, 4],
                                                                   Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)
        
        ccvprModel.setParam("LogToConsole", 0)

        # Select solver.
        if solver == "MH*":
            result = []
            runtime = []
            for n in range(10):
                print("Optimizing " + file[i] + " with MH*: iteration " + str(n + 1))

                t_in = time.time()
                ccvprModel = CCVPR_matheuristic_star(ccvprModel, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I)
                t_fin = time.time()
                
                # Write the solution in a .sol file.
                ccvprModel.write("Results/Large Instances/MHstar/" + file[i] + "_it_" + str(n + 1) + ".sol")

                result = np.append(result, ccvprModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tTime (s): {}".format(round(ccvprModel.ObjVal, 2),
                                                                   round(t_fin - t_in, 2)))
                ccvprModel.reset()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(np.mean(result), 2),
                                         "Time (s)": round(np.mean(runtime), 2)})

        elif solver == "ILS":
            result = []
            runtime = []
            for n in range(10):
                print("Optimizing " + file[i] + " with ILS: iteration " + str(n + 1))

                t_in = time.time()
                ccvprModel = CCVPR_iterated_local_search(ccvprModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
                t_fin = time.time()

                # Write the solution in a .sol file.
                ccvprModel.write("Results/Large Instances/ILS/" + file[i] + "_it_" + str(n + 1) + ".sol")
                
                result = np.append(result, ccvprModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tTime (s): {}".format(round(ccvprModel.ObjVal, 2),
                                                                   round(t_fin - t_in, 2)))
                ccvprModel.reset()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(np.mean(result), 2),
                                         "Time (s)": round(np.mean(runtime), 2)})
    
    # Save the dataframe of the results to a .txt file.
    if solver == "MH*":
        solver = "MHstar"
    df.to_csv("Results/Large Instances/" + solver + "/" + solver + ".txt", sep = ";", index_label = df.columns.name)
    print(df)


# Solve all instances with the specified solver and save the results to .sol and .txt files.
#solve_small_instances("MIP", True)
#solve_small_instances("MIP", True, [])
#solve_small_instances("MH", True)
#solve_small_instances("ILS", True)

#solve_large_instances("MH*")
#solve_large_instances("ILS")

# Import the specified problem instance.
sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax = import_problem_istance("Istances/20 customers/pr01_20.txt")

# Define some missing values of the instance in the following lines.
delta = 60

# The problem instances received don't have data for the max capacity of the vehicles of each carrier Qmax_k.
# We choose to set all the values of Qmax_k to Qmax.
Qmax_k = np.full(sizeK, Qmax)

tu = 0.7
cu = 0.1
#estimate_time_per_unit(K, I, Tmax, tu)
#estimate_cost_per_unit(K, I)

# Obtain the instance model and its variables.
N, sizeN, t, c, Y, X, T, L, Vmin, ccvprModel = CCVPR_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = [1, 3, 4],
                                                           Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)

# Define the running parameters of the MH, MH* and ILS solver.
TLinit = 10
TL = 10
TLpert = 10
Npert = 3
Niter = 5
Nnoimp = 5

#ccvprModel.setParam("LogToConsole", 0);

# Solve using Gurobi MIP solver.
ccvprModel.optimize()
CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MIP solution")
ccvprModel.reset()

# Solve using MH.
ccvprModel = CCVPR_matheuristic(ccvprModel, Y, TLinit, TL, sizeK, sizeI, K, I)
CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MH solution")
ccvprModel.reset()

# Solve using MH*.
ccvprModel = CCVPR_matheuristic_star(ccvprModel, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I)
CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MH* solution")
ccvprModel.reset()

# Solve using ILS.
ccvprModel = CCVPR_iterated_local_search(ccvprModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
CCVPR_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MIP solution")
ccvprModel.reset()

plt.show()