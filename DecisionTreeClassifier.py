import numpy as np
from collections import Counter
from collections import defaultdict


class DecisionTreeClassifier():
    """

    """
    def __init__(self,min_element_perSplit,max_depth=10):
        self.max_depth = max_depth
        self.min_element_perSplit=min_element_perSplit # minimum d'element dans un noeud passé en parametre dans le constructeur

    def build_tree(self,X,Y):
        """
        method pour créer l'arbre : fait appelle a la fonction récursive 'extend_tree' 
        X :  les données 
        Y : les labels des données
        """
        self.number_classes= len(set(Y)) 
        self.number_features = np.shape(X)[1] # dimension du vecteur Xi ( nombre de colonne)
        self.root = self.extend_tree(X,Y,None,0,1) # premmier noeud au sommet on lui passe X,Y et parent_node = None(il n a pas de parent) ,  depth = 0 , un gini = 1 (juste pour initialiser)
        


    def extend_tree(self,x,y,parent_node,depth,parent_gini):
        number_samples = np.shape(x)[0]
        current_node = Node(parent_node)

        if len(set(y)) == 1 : 
            current_node.leaf=True
            current_node.predicted_class = y[0]
            print(y)
            print(len(y))

        elif depth<self.max_depth and number_samples>2*self.min_element_perSplit:    

            feature_index ,split_value,node_gini = self.get_split(x,y,number_samples,parent_gini)

            if feature_index==None :
                current_node.leaf=True
                current_node.predicted_class = (Counter(y).most_common(1))[0][0]
                print(y)
                print(len(y))
                return current_node

            current_node.feature_index, current_node.split_value = feature_index , split_value
            indexes_left,indexes_right = [],[]
            for i,element in enumerate(x): 
                if element[feature_index]>split_value:
                    indexes_left.append(i)
                else:
                    indexes_right.append(i)
            x_left , y_left = x[indexes_left], y[indexes_left]
            x_right, y_right = x[indexes_right], y[indexes_right]
            # print(len(y_left))
            # print(len(y_right))
            depth+=1
            current_node.left=self.extend_tree(x_left,y_left,current_node,depth,node_gini)
            current_node.right=self.extend_tree(x_right,y_right,current_node,depth,node_gini)
        else :
            current_node.leaf=True
            current_node.predicted_class = (Counter(y).most_common(1))[0][0] # right a formula to calculte the majority class in this node 
            print(y)
            print(len(y))
        return current_node

    






    def get_split(self,x,y,number_samples,parent_gini):
        max_iteration = 1000
        def get_gini(dict,nb):
            sum=0
            for d in dict.keys():
                sum+=np.power((dict[d]/nb),2)
            return 1-sum 

        feature = 0
        split_value = 0
        k_iteration=0
        gini = 2
        while(parent_gini-gini<0):
              
            if  k_iteration>max_iteration :
                return None,None,None
            feature = np.random.randint(0,self.number_features)
            f_vector = x[:,feature]
            min_vector = min(f_vector)
            max_vector = max(f_vector)
            split_value = np.random.uniform(min_vector+(min_vector/10),max_vector-(max_vector/10))

            d_left= defaultdict(lambda: 0) 
            d_right= defaultdict(lambda: 0) 
            nb_left=0 
            nb_right = 0

            for i,element in enumerate(x):
                if element[feature] > split_value :
                    d_left[y[i]]+=1
                    nb_left+=1
                else:
                    d_right[y[i]]+=1
                    nb_right+=1

            if nb_left>self.min_element_perSplit and nb_right>self.min_element_perSplit:
                gini = ((nb_left/number_samples)*get_gini(d_left,nb_left))+((nb_right/number_samples)*get_gini(d_right,nb_right))

            k_iteration+=1
        #print(f'k itration :  {k_iteration}')
        return feature,split_value,gini










    def predict(self,X):
        
        def predict_xi(x):
            node = self.root
            while(not (node.leaf)):
                if x[node.feature_index]>node.split_value:
                    node = node.left
                else:
                    node = node.right
                
            return node.predicted_class


        Y = []
        for xi in X :
            Y.append(predict_xi(xi))          
        return Y    



class  Node():
    """
    class qui represente caque noeud 
    """
    def __init__(self,parent_node):
        self.right=None 
        self.left=None
        self.parent = parent_node
        self.leaf=False # boolean pour savoir si ce noeud est une feuille (dernier noeud (il n'a pas de fils))

    






