import numpy as np
from settings import (S_SIZE, A_SIZE, 
                      A_MAX, S_MAX, AREA_DEFEAT, 
                      AREA_WIN, AREA_GENERATION, 
                      TIME, EP_STEPS, S_G_TARG )
import gym
import random
from calculation_scripts.DistanceBW2points import DistanceBW2points
from calculation_scripts.deviation_angle import DeviationAngle as d_ang
import matplotlib.pyplot as plt
import pickle
import socket
import struct
import time

BUFFER_SIZE = 4096
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 6666))# vicon
sock.listen(1)
print('Sock name: {}'.format(sock.getsockname()))
conn, addr = sock.accept()
print('Connected:', addr)
all_data = bytearray()

esp32_ip = '172.20.10.10'  # IP-адрес ESP32 в вашей сети
esp32_port = 111
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((esp32_ip, esp32_port))
timer_ANN = time.perf_counter()

class CustomEnv(gym.Env):
    def __init__(self, target_point=[0,0]):
        self.action_space      = gym.spaces.Box(low=np.array([-A_MAX] * A_SIZE ), high=np.array([A_MAX] * A_SIZE), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([S_MAX  ] * S_SIZE  ), high=np.array([S_MAX ] * S_SIZE ), dtype=np.float32)
        self.__target_point=target_point
        self.reset_env()

    def reset_env(self):
        self.state = np.zeros(S_SIZE) # [position robot_x , position robot_y, 
                                          #  target point X   , target point Y, 
                                          #  angle ]
        self.done = False
        self.__reset_robot()
        self.__x_list, self.__y_list = [], []
        self.__number = 0
        self.__reward = 0
        return self.state

    def __reset_robot(self):
        self.__position_robot = np.zeros(2) # The position of the robot [x, y]
        if self.__target_point != [0,0]:
            self.__target_point =[random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(S_G_TARG, AREA_GENERATION)]
        # while(1):
        #     self.__target_point = [random.uniform(-AREA_GENERATION,AREA_GENERATION), random.uniform(-AREA_GENERATION,AREA_GENERATION)] # Target point [x, y]
        #     if (self.__target_point[0] != 0 or self.__target_point[1] != 0):
        #         break
        self.__x_list, self.__y_list, self.__angle_list = [], [], []    
        self.__old_target_point = self.__target_point
        self.__old_position_robot = 0.0
        
        self.__d_angl_rad = 0.0
        self.__delta_angle = 0.0
        
    def step(self, action):
        # Применяем действие к состоянию и получаем новое состояние
        action = [0.1, action]
        self.__send_to_esp(action[0], action[1])
        self.state = self.__get_state()
        # Проверяем, завершен ли эпизод
        self.done, goal = self.__check_done()
        # Вычисляем награду
        self.__reward = self.__new_reward(goal)
        return self.state, self.__reward, self.done, {}
    
    def __get_state(self):
        data = conn.recv(BUFFER_SIZE)
        # print('Recv: {}: {}'.format(len(data), data))
        data_vicon = pickle.loads(data) # [x_rob, y_rob, 
                                        #  __d_angl_rad]
        print(data_vicon)
        self.__position_robot[0], self.__position_robot[1]= data_vicon[0], data_vicon[1]
        self.__d_angl_rad = data_vicon[2]
        self.__x_list.append(self.__position_robot[0])
        self.__y_list.append(self.__position_robot[1])
        self.__angle_list.append(self.__d_angl_rad)
        return [self.__position_robot[0] , self.__position_robot[1],
                self.__target_point[0]   , self.__target_point[1]  ,
                self.__d_angl_rad ]
    
    @staticmethod
    def __send_to_esp(vel, angl_vel):
        pack_data = struct.pack("ff", vel, angl_vel)
        client_socket.sendall(pack_data)
        
 
    def __new_reward(self, goal):
        if self.done:
            self.__send_to_esp(0, 0)
            if goal:
                return EP_STEPS
            else:
                return -EP_STEPS
        else:
            return self.__dist_reward() + self.__angle_reward() 
            
                       
    def __dist_reward(self):
        dist_new = DistanceBW2points(self.__target_point[0], self.__target_point[1], self.__position_robot[0], self.__position_robot[1]).getDistance()
        dist_old = DistanceBW2points(self.__old_target_point[0], self.__old_target_point[1], self.__old_position_robot[0], self.__old_position_robot[1]).getDistance()
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
                
    def set_number(self,number):
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
        plt.savefig('images_exp/plot' + str(self.__number) + '.png', format='png')
        plt.close()