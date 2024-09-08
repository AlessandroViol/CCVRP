import numpy as np
import pandas as pd
import gurobipy as gb


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

    # C16 (expr. 16): ensures that each carrier's profit (=revenue - travel costs) must be equal or higher than the profit obtainable
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