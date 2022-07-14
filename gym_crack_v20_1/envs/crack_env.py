import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class CrackEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.angle_low = -30.0/180*np.pi # [rad] ###### change this for bigger state & parameter range: -60.0
        self.angle_high = 30.0/180*np.pi # [rad] ###### change this for bigger state & parameter range: 60.0
        self.log_frequency_low = np.log10(10.0) # [Hz] or [1/s]
        self.log_frequency_high = np.log10(100.0) # [Hz] or [1/s] ###### change this for bigger state & parameter range: 1000.0 or 3000.0
        
        act_low = np.array([self.angle_low,
                            self.log_frequency_low,
                           ], dtype=np.float32
        )
        act_high = np.array([self.angle_high,
                             self.log_frequency_high,
                            ], dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low = act_low,
            high = act_high,
            dtype = np.float32
        )
        
        obs_low = np.array([6.0, -40.0/180*np.pi, 0.2], dtype=np.float32) # [6.0, -40.0, 0.1] better ###### change this for bigger state & parameter range: [6.0, -60.0, 0.1 or 0.2]
        obs_high = np.array([8.0, 40.0/180*np.pi, 0.5], dtype=np.float32) # [8.0, 40.0, 0.35] better ###### change this for bigger state & parameter range: [8.0, 60.0, 0.55 or 1.2]
        
        self.observation_space = spaces.Box(
            low = obs_low,
            high = obs_high,
            dtype = np.float32
        )
        
        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        ### update state
        crack_tip = self.crack_tip # [mm, mm]
        a = self.length # [mm] ###### a = self.length or self.state[0]
        Delta_t = self.time # [s]
        sigma = self.sigma # [MPa]
        
        goals = self.goals
        num_goals = self.num_goals
        dists = self.dists # [mm]
        
        a, theta, d = self.state # [a, theta, d] # [mm], [rad], [mm] ###### a = self.length or self.state[0]
        
        beta, log_f = action # [rad], [MPa], [Hz] or [1/s]
        f = 10**log_f # [Hz] or [1/s]
        
        KI = sigma * np.sqrt(np.pi * a/1000) * (np.cos(beta)**2) # [MPa * m^0.5]
        KII = sigma * np.sqrt(np.pi * a/1000) * np.sin(beta) * np.cos(beta) # [MPa * m^0.5]
        
        if KII == 0:
            theta_c = 0 # [rad]
        else:
            theta_c = 2 * np.arctan((KI - np.sqrt(KI**2 + 8*(KII**2))) / (4*KII)) # [rad]
            
        # Assume sigma_min = 0, sigma_max = sigma
        Delta_KI = KI # [MPa * m^0.5]
        Delta_KII = KII # [MPa * m^0.5]
        Delta_Keq = np.sqrt(Delta_KI**2 + Delta_KII**2) # [MPa * m^0.5]
        m = 3.37 # Paris’ exponent
        C = 8.77e-12 # Paris’ constant [(m/cycle) / (MPa * m^0.5)^m]
        Delta_a = C * (Delta_Keq**m) * (Delta_t * f) # [m]
        
        crack_tip_new = crack_tip + 1000 * Delta_a * np.array([np.cos(theta_c), np.sin(theta_c)])
        self.crack_tip = crack_tip_new
        self.length += 1000 * Delta_a # [mm]
        
        dist_new = np.sqrt( (crack_tip_new[0] - goals[self.indicator][0])**2 + (crack_tip_new[1] - goals[self.indicator][1])**2 ) # [mm]
        
        self.steps += 1
        
        ### Reward
        reward = 1 - ( 3 * dist_new / d )**0.2
        reward += 1 - ( abs(theta - theta_c) / (5/180*np.pi) )**0.2
        
        done = False
        if dist_new > d:
            done = True
        if dist_new < 0.1 * d:
            self.indicator += 1
            reward += 10 * self.indicator
        if self.indicator == num_goals:
            done = True
            
        theta_new = np.arctan( (goals[self.indicator][1] - crack_tip_new[1]) / (goals[self.indicator][0] - crack_tip_new[0]) ) #[rad]
        d_new = np.sqrt( (goals[self.indicator][1] - crack_tip_new[1])**2 + (goals[self.indicator][0] - crack_tip_new[0])**2 ) # [mm]
        self.state = np.array([ self.length, theta_new, d_new ]) # [a, theta, d] # [mm], [rad], [mm]
        
        return np.array(self.state), reward, done, {}


    def reset(self):
        crack_tip = np.array([6.0, 0.0]) # initial crack tip is always at (6.0, 0.0)
        self.crack_tip = crack_tip # [mm, mm]
        self.crack_half_length = 6.0 # [mm]
        self.length = self.crack_half_length # [mm]
        self.time = 100.0 # [s]
        self.sigma = 100.0 # [MPa]
        
        # goal point 1
        theta1 = self.np_random.uniform(low = -40.0, high = 40.0) # [degree] ###### change this for bigger state & parameter range: -60, 60
        dist1 = self.np_random.uniform(low = 0.2, high = 0.5) # [mm] ###### change this for bigger state & parameter range: 0.1, 0.55 or 0.2, 1.2
        goal_1_x = 6.0 + dist1 * np.cos(theta1/180*np.pi) # [mm]
        goal_1_y = 0.0 + dist1 * np.sin(theta1/180*np.pi) # [mm]
        goal_1 = np.array([goal_1_x, goal_1_y])
        
        # goal point 2
        theta2 = self.np_random.uniform(low = -40.0, high = 40.0) # [degree] ###### change this for bigger state & parameter range: -60, 60
        dist2 = self.np_random.uniform(low = 0.2, high = 0.5) # [mm] ###### change this for bigger state & parameter range: 0.1, 0.55 or 0.2, 1.2
        goal_2_x = goal_1_x + dist2 * np.cos(theta2/180*np.pi) # [mm]
        goal_2_y = goal_1_y + dist2 * np.sin(theta2/180*np.pi) # [mm]
        goal_2 = np.array([goal_2_x, goal_2_y])
        
        # goal point 3
        theta3 = self.np_random.uniform(low = -40.0, high = 40.0) # [degree] ###### change this for bigger state & parameter range: -60, 60
        dist3 = self.np_random.uniform(low = 0.2, high = 0.5) # [mm] ###### change this for bigger state & parameter range: 0.1, 0.55 or 0.2, 1.2
        goal_3_x = goal_2_x + dist3 * np.cos(theta3/180*np.pi) # [mm]
        goal_3_y = goal_2_y + dist3 * np.sin(theta3/180*np.pi) # [mm]
        goal_3 = np.array([goal_3_x, goal_3_y])
        
        # goal point 4
        theta4 = self.np_random.uniform(low = -40.0, high = 40.0) # [degree] ###### change this for bigger state & parameter range: -60, 60
        dist4 = self.np_random.uniform(low = 0.2, high = 0.5) # [mm] ###### change this for bigger state & parameter range: 0.1, 0.55 or 0.2, 1.2
        goal_4_x = goal_3_x + dist4 * np.cos(theta4/180*np.pi) # [mm]
        goal_4_y = goal_3_y + dist4 * np.sin(theta4/180*np.pi) # [mm]
        goal_4 = np.array([goal_4_x, goal_4_y])
        
        # goal point 5
        theta5 = self.np_random.uniform(low = -40.0, high = 40.0) # [degree] ###### change this for bigger state & parameter range: -60, 60
        dist5 = self.np_random.uniform(low = 0.2, high = 0.5) # [mm] ###### change this for bigger state & parameter range: 0.1, 0.55 or 0.2, 1.2
        goal_5_x = goal_4_x + dist5 * np.cos(theta5/180*np.pi) # [mm]
        goal_5_y = goal_4_y + dist5 * np.sin(theta5/180*np.pi) # [mm]
        goal_5 = np.array([goal_5_x, goal_5_y])
        
        # pseudo goal point, useless
        goal_pseudo = np.array([99.99, 0.0])
        
        self.goals = [goal_1, goal_2, goal_3, goal_4, goal_5, goal_pseudo]
        self.num_goals = 5
        self.dists = [dist1, dist2, dist3, dist4, dist5] # [mm]
        
        self.indicator = 0  # indicates which of the goal points should the crack growth aims
        self.away = False # if the crack is going away from the goal point
        self.steps = 0 # steps for this episode
        self.steps_taken = 0 # steps taken to reach the current goal point
        self.reached_last_point = False # whether the last goal point is reached or not
        
        self.state = np.array([ self.length, theta1/180*np.pi, dist1 ]) # [a, theta, d] # [mm], [rad], [mm]
        
        return self.state


    def render(self, mode='human'):
        pass


    def close(self):
        pass
