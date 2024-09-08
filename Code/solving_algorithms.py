import random
from itertools import combinations


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
