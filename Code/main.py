import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from CCVRP_model import CCVRP_model
from CCVRP_instances import import_problem_istance
from CCVRP_instances import estimate_cost_per_unit
from CCVRP_instances import estimate_time_per_unit
from solving_algorithms import CCVRP_matheuristic
from solving_algorithms import CCVRP_matheuristic_star
from solving_algorithms import CCVRP_iterated_local_search


# Return a plot of the routes for each period and each carrier for the provided solved model. Original carriers and depots are marked as well.
def CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X):
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
        N, sizeN, t, c, Y, X, T, L, Vmin, ccvrpModel = CCVRP_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = vi_indexes,
                                                                   Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)
        ccvrpModel.setParam("LogToConsole", 0)

        # Select solver.
        if solver == "MIP":
            print("Optimizing " + file[i] + " with MIP")

            ccvrpModel.setParam("TimeLimit", 10 * 60)
            ccvrpModel.optimize()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(ccvrpModel.ObjVal, 2), 
                                         "Gap (%)": round(ccvrpModel.MIPGap * 100, 2), 
                                         "Time (s)": round(ccvrpModel.Runtime, 2)})

            # Write the solution in a .sol file.
            if vi_indexes == []:
                ccvrpModel.write("Results/Small Instances/MIP/" + file[i] + "_no_VI.sol")
            else:
                ccvrpModel.write("Results/Small Instances/MIP/" + file[i] + ".sol")

            # Create a plot of the solution routing.
            if save_fig:
                CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
                plt.savefig("Results/Small Instances/MIP/" + file[i] + ".png")

            print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvrpModel.ObjVal, 2), 
                                                                        round(ccvrpModel.MIPGap * 100, 2), 
                                                                        round(ccvrpModel.Runtime, 2)))

        elif solver == "MH":
            print("Optimizing " + file[i] + " with MH")

            # Import the BKS obtained from the MIP to compute the relative gap.
            BKS = pd.read_csv("Results/Small Instances/MIP/MIP.txt", sep=";")
            BKS = BKS["Solution Value"][i]

            t_in = time.time()
            ccvrpModel = CCVRP_matheuristic(ccvrpModel, Y, TLinit, TL, sizeK, sizeI, K, I)
            t_fin = time.time()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(ccvrpModel.ObjVal, 2),
                                         "Gap (%)": round((ccvrpModel.ObjVal - BKS) * 100 / BKS, 2), 
                                         "Time (s)": round(t_fin - t_in, 2)})

            # Write the solution in a .sol file.
            ccvrpModel.write("Results/Small Instances/MH/" + file[i] + ".sol")

            print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvrpModel.ObjVal, 2), 
                                                                        round((ccvrpModel.ObjVal - BKS) * 100 / BKS, 2), 
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
                ccvrpModel = CCVRP_iterated_local_search(ccvrpModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
                t_fin = time.time()

                # Write the solution in a .sol file.
                ccvrpModel.write("Results/Small Instances/ILS/" + file[i] + "_it_" + str(n + 1) + ".sol")
                
                result = np.append(result, ccvrpModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tGap (%): {}\tTime (s): {}".format(round(ccvrpModel.ObjVal, 2), 
                                                                            round((ccvrpModel.ObjVal - BKS) * 100 / BKS, 2), 
                                                                            round(t_fin - t_in, 2)))
                ccvrpModel.reset()
            
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
        N, sizeN, t, c, Y, X, T, L, Vmin, ccvrpModel = CCVRP_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = [1, 3, 4],
                                                                   Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)
        
        ccvrpModel.setParam("LogToConsole", 0)

        # Select solver.
        if solver == "MH*":
            result = []
            runtime = []
            for n in range(10):
                print("Optimizing " + file[i] + " with MH*: iteration " + str(n + 1))

                t_in = time.time()
                ccvrpModel = CCVRP_matheuristic_star(ccvrpModel, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I)
                t_fin = time.time()
                
                # Write the solution in a .sol file.
                ccvrpModel.write("Results/Large Instances/MHstar/" + file[i] + "_it_" + str(n + 1) + ".sol")

                result = np.append(result, ccvrpModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tTime (s): {}".format(round(ccvrpModel.ObjVal, 2),
                                                                   round(t_fin - t_in, 2)))
                ccvrpModel.reset()

            # Save the result in the appropriate row of the dataframe.
            df.loc[file[i]] = pd.Series({"Solution Value": round(np.mean(result), 2),
                                         "Time (s)": round(np.mean(runtime), 2)})

        elif solver == "ILS":
            result = []
            runtime = []
            for n in range(10):
                print("Optimizing " + file[i] + " with ILS: iteration " + str(n + 1))

                t_in = time.time()
                ccvrpModel = CCVRP_iterated_local_search(ccvrpModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
                t_fin = time.time()

                # Write the solution in a .sol file.
                ccvrpModel.write("Results/Large Instances/ILS/" + file[i] + "_it_" + str(n + 1) + ".sol")
                
                result = np.append(result, ccvrpModel.ObjVal)
                runtime = np.append(runtime, t_fin - t_in)

                print("\tSolution Value: {};\tTime (s): {}".format(round(ccvrpModel.ObjVal, 2),
                                                                   round(t_fin - t_in, 2)))
                ccvrpModel.reset()

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
N, sizeN, t, c, Y, X, T, L, Vmin, ccvrpModel = CCVRP_model(sizeK, sizeI, sizeP, P, K, I, Tmax, Qmax, tu, cu, delta, vi_indexes = [1, 3, 4],
                                                           Qmax_k = Qmax_k, selfConstr = True, TConstr = True, LConstr = True)

# Define the running parameters of the MH, MH* and ILS solver.
TLinit = 10
TL = 10
TLpert = 10
Npert = 3
Niter = 5
Nnoimp = 5

#ccvrpModel.setParam("LogToConsole", 0);

# Solve using Gurobi MIP solver.
ccvrpModel.optimize()
CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MIP solution")
ccvrpModel.reset()

# Solve using MH.
ccvrpModel = CCVRP_matheuristic(ccvrpModel, Y, TLinit, TL, sizeK, sizeI, K, I)
CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MH solution")
ccvrpModel.reset()

# Solve using MH*.
ccvrpModel = CCVRP_matheuristic_star(ccvrpModel, Y, TLinit, TL, Nnoimp, sizeK, sizeI, K, I)
CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MH* solution")
ccvrpModel.reset()

# Solve using ILS.
ccvrpModel = CCVRP_iterated_local_search(ccvrpModel, Y, sizeK, sizeI, K, I, TLinit, TL, TLpert, Nnoimp, Niter, Npert)
CCVRP_plot_route(sizeK, sizeP, sizeN, P, I, N, X)
plt.subplot(int(np.ceil(sizeP / 2)), int(np.floor(sizeP / 2)), 1)
plt.title("MIP solution")
ccvrpModel.reset()

plt.show()