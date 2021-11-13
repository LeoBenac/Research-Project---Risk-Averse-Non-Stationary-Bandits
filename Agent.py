

import numpy as np

class Agent:
    def __init__(self, c_values, nbr_arms=10, estimators=np.zeros((10,)), time_steps=1000):
        """ 
        This Class was created to assist my numerical experiments performed during my research internship
        on Risk Averse Non-Stationary Multi Armed Bandits. This class contains a method 'run_shared_experience'
        that will run the same experiment using 3 different algorithms simultaneously to refine each arm's CVaR estimates.
        
        Agent Class for k-armed bandit problems.

        Parameters
        ---------
        
        c_values : numpy array
            grid of values of c that will be use to estimate our CVaR for the dual recursive method
            
        nbr_arms : int
            number of k arms in the bandit problem
            
        estimators : numpy array
            initial estimator of the mean of the distribution of each arms at initial time
            
        time_steps : int
            number of steps allowed

        
         Attributes
         ----------
         
         k : int
             number of arms
             
         max_steps : int
             number of time steps allowed
             
         action_taken_i : numpy array
             action taken for the 3 different estimation methods
        
         rewards_received_i : numpy array
             rewards received during the experiment for the 3 different estimation methods
            
         t : int
             current time step
             
        reward_record_i : list of integers
            keep track of the rewards for the Sample averaging and Weighted empirical methods 
            as the Dual recursive does not need to store all of the rewards
            
        CVRa_values_i : numpy array
            keep track of the CVaR estimates for the Sample averaging and Weighted empirical methods 
            
        q_a_values_i : numpy array
            keep track of the VaR estimates for the Sample averaging and Weighted empirical methods 

        c_values : numpy array
            grid of values of c that will be use to estimate our CVaR for the dual recursive method
        
        z : numpy array
            2D numpy array where z[i,j]= the estimate of the CVaR of the ith arms using the jth c value
            
        CVaR_actual_best: list of numbers
            contains the true values of the CVaR of the best arm at each time step t
            
        CVaR_actual_best_actions: list of integers
            contains the true best actions at each time step t
            
        CVaR_best_selected_i : list of numbers
            contains the true values of the CVaR of the arm selected at time step t 
            for the 3 different estimation methods

         Methods
         -------
         
         run_shared_experience(alpha, stepsize, rewards, CVaRs)
             alpha is the threshold value for our CVaR estimates, 
             stepsize is an array of numbers to compare the rate of learning of our algorithms.
            
             A 2D array of rewards for each arms across all time steps as well as the 
             true CVaRs for each arms for each time step are generated before we run our method
             so that each estimation methods can be used on the same data in order to reduce
             variance in our results.
        """

        self.k = nbr_arms
        self.max_steps = time_steps
        self.t = 0

        self.action_taken_1 = np.zeros((time_steps,)) - 1
        self.action_taken_2 = np.zeros((time_steps,)) - 1
        self.action_taken_3 = np.zeros((time_steps,)) - 1

        self.reward_received_1 = []
        self.reward_received_2 = []
        self.reward_received_3 = []

        self.reward_record_1 = [[] for i in range(nbr_arms)]
        self.reward_record_2 = [[] for i in range(nbr_arms)]
        
        self.CVRa_values_1 = np.array(estimators) - 0.0
        self.CVRa_values_2 = np.array(estimators) - 0.0

        self.q_a_values_1 = np.array(estimators) + 0.0
        self.q_a_values_2 = np.array(estimators) + 0.0

        self.c_values = c_values
        self.z = [list(np.zeros(len(c_values)) - 0.0) for _ in range(self.k)]

        self.CVaR_actual_best = []
        self.CVaR_actual_best_actions = []
        self.CVaR_best_selected_1 = []
        self.CVaR_best_selected_2 = []
        self.CVaR_best_selected_3 = []


    def run_shared_experience(self, alpha, stepSize, rewards, CVaRs, epsilon = 0.05 ):
        
        """
        The agent will run the 3 different estimation methods simultaneously, 
        where the simulated rewards with corresponding CVaR estimate for each arms
        has been generated already to reduce variance among final results.
        
        Parameters:
        ----------
        
        alpha : int
            alpha threshold for our CVaR estimates
        
        stepSize : numpy array
            Array of different stepsizes (learning rate) to get a better idea of how it affects 
            our estimations methods
            
        rewards: numpy array
            2D numpy array of rewards generated before the method is called
            where rewards[i,t] = the reward of arm i at time t
        
        CVaRs : numpy array
            2D numpy array of the true CVaR generated before the method is called
            where CVaRs[i,t] = the true CVaR of arm i at time t. Note that
            the true CVaRs are constantly changing since we are in a non-stationary context
        
        epsilon : number
            control the degree of exploration of our epsilon-greedy policy
        """

        a_1 = [min(i) for i in self.z]
        best_arm_1 = (np.random.choice((np.where((min(a_1) == a_1))[0])))

        a_2 = self.CVRa_values_1
        best_arm_2 = (np.random.choice((np.where((min(a_2) == a_2))[0])))

        a_3 = self.CVRa_values_2
        best_arm_3 = np.argmin(a_3)

        explore = (np.random.random())

        if explore < epsilon:
            best_arm_1 = np.random.randint(self.k)
            best_arm_2 = np.random.randint(self.k)
            best_arm_3 = np.random.randint(self.k)

        self.action_taken_1[self.t] = best_arm_1
        self.action_taken_2[self.t] = best_arm_2
        self.action_taken_3[self.t] = best_arm_3

        self.t += 1

        self.CVaR_best_selected_1.append(CVaRs[best_arm_1][self.t - 1])
        self.CVaR_best_selected_2.append(CVaRs[best_arm_2][self.t - 1])
        self.CVaR_best_selected_3.append(CVaRs[best_arm_3][self.t - 1])

        reward_1 = rewards[best_arm_1][self.t - 1]
        reward_2 = rewards[best_arm_2][self.t - 1]
        reward_3 = rewards[best_arm_3][self.t - 1]

        self.reward_received_1.append(reward_1)
        self.reward_received_2.append(reward_2)
        self.reward_received_3.append(reward_3)

        step_size = stepSize


        #Dual Recursive 

        target_1 = np.array(self.c_values)
        target_1[reward_1 > target_1] = target_1[reward_1 > target_1] + (
                    (1 / (1 - alpha)) * (reward_1 - target_1[reward_1 > target_1]))
        past_estimate_1 = np.array(self.z[best_arm_1])
        new_estimate_1 = (past_estimate_1 + (step_size * (target_1 - past_estimate_1)))
        self.z[best_arm_1] = new_estimate_1


        #Sample averaging 

        self.reward_record_1[best_arm_2].append(reward_2)

        a1 = sorted(self.reward_record_1[best_arm_2], reverse=True)

        p_1 = 1. * (np.arange(len(a1)) + 1) / len(a1)
        self.q_a_values_1[best_arm_2] = a1[np.where(p_1 >= (1 - alpha) )[0][0]]

        check_1 = a1 < self.q_a_values_1[best_arm_2]

        if (np.where(check_1 == True)[0]).size == 0:
            ind_1 = 0
        else:
            ind_1 = (np.where(check_1 == True)[0][0] - 1)

        temp_1 = a1[:ind_1 + 1]

        self.CVRa_values_1[best_arm_2]  = (sum(temp_1) / len(temp_1))


        #Weighted empirical 

        self.reward_record_2[best_arm_3].append(reward_3)

        a2 = (self.reward_record_2[best_arm_3])

        w_3 = np.array(
            ((1-step_size) ** (len(a2) - (np.arange(len(a2)) + 1))) * ((1 - (1-step_size)) / (1 - ( (1-step_size) ** (len(a2)))) ))
        w_3[w_3 < 1e-20] = 0

        d3_ = np.array(list(zip(np.arange(len(self.reward_record_2[best_arm_3])) + 1, self.reward_record_2[best_arm_3], w_3)))

        s_3 = np.array(sorted(d3_[:, 1:], key=lambda x: x[0], reverse=False))

        F_s_3 = np.array(list(zip(np.cumsum(s_3[:, 1]), s_3[:, 0])))[::-1]

        ind_3 = np.where(np.array(F_s_3)[:, 0] >= alpha)[0][-1]
        q_a_3 = (np.array(F_s_3)[:, 1][ind_3])

        self.q_a_values_2[best_arm_3] = q_a_3

        CVaR_3 = (sum((d3_[:, 1] * d3_[:, 2])[d3_[:, 1] >= q_a_3]) / sum(d3_[:, 2][d3_[:, 1] >= q_a_3]))

        self.CVRa_values_2[best_arm_3] = CVaR_3

def get_empirical_CVaR(rewards, alpha = 0.9):
    
    """
    Helper method (not part of any class) used to calculate the empirical CVar
    of a given sequence of rewards for a specific alpha threshold
    
    Parameters
    ----------
    
    rewards : numpy array
        array of reward values
    
    alpha : number
        threshold to calculate empirical CVaR using a default of 0.1
    
    """

    a = sorted(list(rewards).copy(), reverse= True)

    p = 1. * (np.arange(len(a)) + 1) / len(a)
    q_a = a[np.where(p >= (1 - alpha) )[0][0]]

    check = a < q_a

    if (np.where(check == True)[0]).size == 0:
        ind = 0
    else:
        ind = (np.where(check == True)[0][0] - 1)

    temp = a[:ind + 1]

    return (sum(temp) / len(temp))