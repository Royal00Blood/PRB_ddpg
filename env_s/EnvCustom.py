import numpy as np
from settings import (S_SIZE, A_SIZE, A_MAX, S_MAX, AREA_DEFEAT, 
                      AREA_WIN, AREA_GENERATION, TIME, S_G_TARG, REWARD, EP_STEPS)
import gym
import random
from calculation_scripts.DistanceBW2points import DistanceBW2points as d_dist
from calculation_scripts.deviation_angle import DeviationAngle as d_ang
import matplotlib.pyplot as plt

class CustomEnv(gym.Env):
    def __init__(self,target_point = [0,0]):
        self.action_space      = gym.spaces.Box(low=np.array([-A_MAX] * A_SIZE ), high=np.array([A_MAX] * A_SIZE), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-S_MAX] * S_SIZE ), high=np.array([A_MAX] * S_SIZE), dtype=np.float32)
        self.__target_point= target_point
        self.reset_env()

    def reset_env(self):
        self.state = np.zeros(S_SIZE) # [position robot_x , position robot_y, 
                                          #  target point X   , target point Y, 
                                          # angle]
        self.done = False
        self.__reset_robot()
        self.__x_list, self.__y_list, self.__angle_list = [], [], []
        self.__number = 0
        self.__reward = 0
        return self.state

    def __reset_robot(self):
        self.__position_robot = np.zeros(2) # The position of the robot [x, y]
        self.__robot_quat     = np.zeros(4) # Quaternions [Qx, Qy, Qz, Qw]
        # self.__target_point =[random.choice([random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(-AREA_GENERATION,-S_G_TARG)]),
        #                       random.choice([random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(-AREA_GENERATION,-S_G_TARG)])]
        # if self.__target_point != [0,0]:
        self.__target_point =[random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(S_G_TARG, AREA_GENERATION)]    
        self.__old_target_point = self.__target_point
        self.__old_position_robot = [0,0]
        
        self.__d_angl_rad = 0.0
        self.__delta_angle = 0.0
        
    def step(self, action):
        # Применяем действие к состоянию и получаем новое состояние
        # action [v, w]
        self.state = self.__calculate_state(action)
        # Проверяем, завершен ли эпизод
        goal = self.__check_done()
        # Вычисляем награду
        self.__reward = self.__new_reward(goal)
        return self.state, self.__reward, self.done, {}
    
    def __calculate_state(self,action):
        # action [v, w] 
        v = action[0]
        w = action[1]
        self.__d_angl_rad += (w * TIME)  # изменение угла в радианах
        
        # The position of the robot [x, y]
        self.__position_robot[0] += v * np.cos(self.__d_angl_rad) * TIME
        self.__position_robot[1] += v * np.sin(self.__d_angl_rad) * TIME
        
        # Quaternions [Qx, Qy, Qz, Qw]
        self.__robot_quat[0] = 0.0
        self.__robot_quat[1] = 0.0
        self.__robot_quat[2] = 1 * np.sin(self.__d_angl_rad / 2)
        self.__robot_quat[3] = np.cos(self.__d_angl_rad / 2)
        self.__x_list.append(self.__position_robot[0])
        self.__y_list.append(self.__position_robot[1])
        self.__angle_list.append(self.__d_angl_rad)
        return [self.__position_robot[0] , self.__position_robot[1],
                self.__target_point[0]   , self.__target_point[1]  ,
                self.__d_angl_rad]
 
    def __new_reward(self, goal):
        if self.done:
            if goal:
                self.graf_move()
                return REWARD
            else:
                return -REWARD
        else:
            return self.__dist_reward() + self.__angle_reward() - self.__number/EP_STEPS
                       
    def __dist_reward(self):
        dist_new = d_dist(self.__target_point[0], self.__target_point[1], self.__position_robot[0], self.__position_robot[1]).getDistance()
        dist_old = d_dist(self.__old_target_point[0], self.__old_target_point[1], self.__old_position_robot[0], self.__old_position_robot[1]).getDistance()
        self.__old_target_point = [self.__target_point[0], self.__target_point[1]]
        self.__old_position_robot = [self.__position_robot[0], self.__position_robot[1]]
        return dist_old - dist_new 
           
    def __angle_reward(self):
        self.__delta_angle = d_ang(self.__position_robot[0], self.__position_robot[1],
                                   self.__target_point[0], self.__target_point[1],
                                   self.__d_angl_rad
                                   ).get_angle_dev()
        return -abs(self.__delta_angle)
              
    def __check_done(self):
        goal = False
        if( - AREA_DEFEAT > self.__position_robot[0] or self.__position_robot[0] > AREA_DEFEAT  or
            - AREA_DEFEAT > self.__position_robot[1] or self.__position_robot[1] > AREA_DEFEAT):
            goal = False
            self.done = True
            return goal
        elif ( self.__target_point[0] + AREA_WIN > self.__position_robot[0] > self.__target_point[0] - AREA_WIN   and
               self.__target_point[1] + AREA_WIN > self.__position_robot[1] > self.__target_point[1] - AREA_WIN):
            goal = True
            self.done = True
            #self.graf_move()
            print("Target_True")
            return  goal
        else:
            self.done = False
            return goal
                
    def set_number(self, number):
        self.__number = number
           
    def graf_move(self):
        
        plt.figure(self.__number)
        # plot x(y)
        plt.title("The trajectory of the robot")
        plt.plot(self.__x_list, self.__y_list)
        plt.plot(0, 0, marker="o", color='green')
        plt.plot(self.__target_point[0], self.__target_point[1], color='red')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('images/plot' + str(self.__number) + '.png', format='png')
        plt.close()