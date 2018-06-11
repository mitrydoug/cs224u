import random
from math import floor, inf

import numpy as np 

"""
Philosophical musings:

Unrelated sparse users clustered together because of a bridge super user:
    Question: should we have the centroid update have a threshold for minimum number of times that a movie is mentioned?
    Motivation:
        Right now if we have the centers take in every single rating, centers are a huge superset of all ratings of it's users
        We have some users who rated a ton of products and some users who have barely rated any
        Let's say user1 rated movies 1-10
        and user2 rated movies 40-50
        If some power user rated movies 1-50 and also is in the same cluster and influencing the centroid center
            user1 and user2 could find themselves sticky in the same cluster, although they have nothing in common
        Is this bad? is this good? is the alternative having a bunch of unassined users? (obviously bad)

"""



class ClusteringModel():
    class user():
        # This is a class for a single user
        # include
        #     user_id: [type: int] [description: same user id as in original data] 
        #     assigned_cluster
        #     ratings: dictionary of ((int)product_id, (float)rating) that includes all rating given by object

        # inputs:
        #    user_id (int)
        #    grouped_data: a pandas dataframe with data only for the user includes the colums
        #       product_id: The rated product ids
        #       rating: the corresponding ratings 
        def __init__(self,user_id, grouped_data):
            self.user_id = user_id
            self.assigned_cluster = None
            
            product_ids = grouped_data['product_id'].tolist()
            rating_list = grouped_data['rating'].tolist()
            # assert len(product_ids) == len(rating_list)
            self.ratings = {product_ids[i]:rating_list[i] for i in range(len(product_ids))}

    class centroid():
       # The class for the centroids
        # include 
        #     cluster_id: [type: int] [description: an identifier for the centroid]
        #     center: dictionary where keys are a product_id  and values are associated
        #     new_center: dictionary being built up as users are being added in
        #         will be used to construct the center of the next iteration
        #         key = product_id
        #         val = (sum, num) where (float)sum is sum from the users, (float)num is # of times has been updated

        def __init__(self,centroid_id, initial_center):
            self.centroid_id = centroid_id
            self.center = initial_center
            self.new_center = {}

        # assumes u is of type user, as defined above  
        def add_user(self, u):
            for product_id in u.ratings:
                rating = u.ratings[product_id]
                if product_id in self.new_center:
                    new_sum = self.new_center[product_id][0] + rating
                    new_count = self.new_center[product_id][1] + 1
                    self.new_center[product_id] = (new_sum, new_count)
                else:
                    self.new_center[product_id] = (rating, 1)
        
        def recenter(self):
            self.center = {}
            self.center = {k:(self.new_center[k][0]/self.new_center[k][1]) for k in self.new_center.keys()}
            self.new_center = {}
        
    """
    algorithm

    Note: this algorithm expects the data to already be normalized
        i.e. The input data has already been mean centered etcetera 
        
    Set up: 
    1) groups <- break up data frame into groups (note can trash dataframe at this point to save memory space)
    2) create a list/vector of users:
        from each group create a user object (note can trash groups here to save memory space)
    3) initialize centroid objects, potentially as randomly selected users
    
    fit: alternate between
    1) Assigning users to groups
        For each user u:
            Find the centroid closest to u by testing against all centroids
            Add to u to the centroid (see centroid add_user function)
    2) Recenter each centroid 
    """
    # This is the meat of the implmentation class for K means
    # instance variables
    #      num_clusters <int>: number of centroids generated
    #      num_iterations <int>: the number of  
    #      users: list of all users 
    #      clusters: list of all clusters 
    def __init__(self, num_clusters=100,num_iterations=1000,**kwargs):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations 

    # The meat of this algorithm, where the most important decisions are made 
    # inputs: assumes that users and centroids are instances of the classes above
    # output: <double> the similarity between the user and 
    def _distance(self, user, centroid):
        # centroid center: dictionary where keys are a product_id  and values are associated
        # user ratings: list of ((int)product_id, (float)rating) tuples that includes all rating given by object
        
        set_centroid = set(centroid.center.keys())
        set_user =  set(user.ratings.keys())
       
        # intersection only 
        intersection = set_centroid & set_user
                
        dif = np.array([(user.ratings[k] - centroid.center[k])**2 for k in intersection]) 
        l2_intersection = np.sqrt(np.sum(dif)) 
        
        # centroid only 
        just_centroid = set_centroid - intersection
        l2_centroid = np.sqrt(np.sum([centroid.center[k]**2 for k in just_centroid]))

        # user only 
        just_user = set_user - intersection
        l2_user = np.sqrt(np.sum(np.array([user.ratings[k]**2 for k in just_user])))
        
        USER_FACTOR = 1
        CENTROID_FACTOR = .01
        
        return l2_intersection + (l2_user * USER_FACTOR) +  (l2_centroid * CENTROID_FACTOR)
    
    
    def _assign_all_users(self):
        # note for now, if similarity is zero to all centers
        #    keep unassigned and hope that another user will tie the two together 

        for u in self.users:
            min_distance = inf
            u.assigned_cluster = -1 # unassigned
            for c in self.clusters:
                if self._distance(u,c) < min_distance:
                    u.assigned_cluster = c.centroid_id
                    min_distance = self._distance(u,c)
                    c.add_user(u)

    def _initialize_users(self, X):
        grouped_reviews = X['user_product_ratings'].groupby('user_id')['user_id', 'product_id', 'rating']
        group_names = grouped_reviews.groups.keys()
        self.users = [self.user(g, grouped_reviews.get_group(g)) for g in group_names]

    def _initialize_clusters(self):
        gen_rand_usr = lambda : self.users[floor(random.uniform(1, len(self.users)))]
        self.clusters = [self.centroid(c, gen_rand_usr().ratings)\
            for c in range(self.num_clusters)]

    def fit(self, X):
        self._initialize_users(X)
        self._initialize_clusters()
        
        for i in range(self.num_iterations):
            # assign all users
            self._assign_all_users()
            # center all clusters
            for c in self.clusters:
                c.recenter()
            
    # will return some kind of list or mapping between each user id and what cluster they belong to.
    # note: cluster = -1 means unassigned
    def return_clusters(self):
        return {u.user_id: u.assigned_cluster for u in self.users}