import math
import numpy as np
from numpy import array
from numpy import linalg as LA
def get_knn(train_data, test_point, k):

    knn_d = []
    knn = []
    distance = train_data - test_point
    for idx, point in enumerate(distance):
        point = distance[idx,1:85]
        point_dst = LA.norm(point)
        point_dst = (np.abs(point_dst))
        if math.isnan(point_dst):
            continue
        if len(knn) != k:
            knn.append(idx)
            knn_d.append(point_dst)
        else:
            furthest_n = (np.abs(knn_d)).argmax()
            if(point_dst < knn_d[furthest_n] or knn_d[furthest_n] == 0):
                knn_d[furthest_n] = point_dst
                knn[furthest_n] = idx

    return knn

def cross_validation(kfold_data):
    
    k = 2000
    single_result = []
    overall_results = []
    train_set = []
    empty = 1
    for idx_t, test_set in enumerate(kfold_data):
        for idx_s, subsets in enumerate(kfold_data):
            if empty == 1 and idx_t != idx_s:
                train_set = np.array(subsets)
                empty = 0
            elif idx_t !=idx_s:
                train_set = np.vstack((train_set, subsets))

        for idx, point in enumerate(test_set):
            
            go_beavs = 0
            knn = get_knn(train_set, point, k)
            knn.sort()
            for i in range(k):
                x = knn[i]
                if train_set[x][86] == 1:
                    
                    go_beavs += 1
            
            if go_beavs > ((k/2)):
                if point[86] == 1:
                    single_result.append(1)
                else:
                    single_result.append(0)
            else: 
                if point[86] == 1:
                    single_result.append(0)
                else:
                    single_result.append(1) 
        overall_results.append(np.average(single_result))
        single_result.clear()
        print(overall_results)
        print("Mean = ",np.average(overall_results))
        print("Variance = ",np.var(overall_results))

        empty = 0
    return np.average(overall_results)         

def prediction(train_data, test_data, k):
    r = open("results.csv", "a")
    r.write("id,income\n")
    train_set = train_data[ : ,1:86]
    test_set = test_data[ : ,1:86]
    result = []
    for idx, point in enumerate(test_set):
            
        go_beavs = 0
        knn = get_knn(train_set, point, k)
        knn.sort()
        for i in range(k):
            x = knn[i]
            if train_data[x][86] == 1:
                    
                go_beavs += 1
            
        if go_beavs > ((k/2)):
            result=[test_data[idx][0], 1]
        else: 
            result=[test_data[idx,0], 0]
        line = str(int(result[0]))+","+str(result[1])+"\n"
        r.write(line) 
    r.close()
    return result
def main():
    kfold = 4
    k = 7
    with open('train.csv') as train_file:

        train_data = np.loadtxt(train_file, delimiter = ',', skiprows = 1, )
        kfold_data = np.vsplit(train_data, kfold)
    with open('test_pub.csv') as test_file:

        test_data = np.loadtxt(test_file, delimiter = ',', skiprows = 1, )

    #result = prediction(train_data, test_data, 55)
    #print()
    accuracy = cross_validation(kfold_data)
    

    


main()