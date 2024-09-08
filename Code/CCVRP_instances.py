import numpy as np
import pandas as pd


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