import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    idx = X.shape[0]
    random_k_indices = np.random.choice(idx, k, replace=False)
    centroids = X[random_k_indices, :]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_instances = X.shape[0]
    distances = np.zeros((centroids.shape[0], num_instances))
    # i up 7 , y= (x,y,z)
    for i, y in enumerate(centroids):
        distances[i] = np.sum(np.abs(X - y)**p, axis=1)**(1/p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
#     print(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    prev_centroids = centroids.copy()

    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = []

        for j in range(k):
            cent = np.mean(X[classes == j], axis=0)
            new_centroids.append(cent)

        centroids = np.array(new_centroids)

        if np.array_equal(centroids, prev_centroids):
            break
        else:
            prev_centroids = centroids
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

# def kmeans_pp(X, k, p ,max_iter=100):
#     """
#     Your implenentation of the kmeans++ algorithm.
#     Inputs:
#     - X: a single image of shape (num_pixels, 3).
#     - k: number of centroids.
#     - p: the parameter governing the distance measure.
#     - max_iter: the maximum number of iterations to perform.

#     Outputs:
#     - The calculated centroids as a numpy array.
#     - The final assignment of all RGB points to the closest centroids as a numpy array.
#     """
#     classes = None
#     centroids = None
#     ###########################################################################
#     # TODO: Implement the function.                                           #
#     ###########################################################################
#     pass
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return centroids, classes



import sys

def distance(p1,p2):
    return np.sum((p1 - p2)**2)
    

def initialize(X, k):
    
    centroids = []
    idx = X.shape[0]
    rand_idx = np.random.choice(idx, 1, replace=False)
    centroids.append(X[rand_idx,:])
#     print(centroids)
    
    for c_id in range(k-1):
        
        dist = []
        for i in range(X.shape[0]):
            point = X[i,:]
            d = sys.maxsize
            
            for j in range(len(centroids)):
                temp_dist = distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
            
        dist = np.array(dist)
        dist = dist/np.sum(dist)
        next_centroid = X[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
    return np.asarray(centroids)



## oren kmeans pp:
def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = initialize(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    prev_centroids = centroids.copy()

    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        new_centroids = []

        for j in range(k):
            cent = np.mean(X[classes == j], axis=0)
            new_centroids.append(cent)

        centroids = np.array(new_centroids)

        if np.array_equal(centroids, prev_centroids):
            break
        else:
            prev_centroids = centroids
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes




