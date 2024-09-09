import numpy as np
from settings import (STATE_SIZE, ACTION_SIZE, 
                      ACTION_, STATE_, AREA_DEFEAT, 
                      AREA_WIN, AREA_GENERATION, 
                      TIME, EP_STEPS )
import gym
import random
from DistanceBW2points import DistanceBW2points
from deviation_angle import DeviationAngle as d_ang
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
    def __init__(self):
        self.action_space      = gym.spaces.Box(low=np.array([-ACTION_] * ACTION_SIZE ), high=np.array([ACTION_] * ACTION_SIZE), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([STATE_  ] * STATE_SIZE  ), high=np.array([STATE_ ] * STATE_SIZE ), dtype=np.float32)
        self.reset_env()

    def reset_env(self):
        self.state = np.zeros(STATE_SIZE) # [position robot_x , position robot_y, 
                                          #  target point X   , target point Y, 
                                          #  velocity         , angular velocity, 
                                          #  Quaternions Z    , Quaternions W ]
        self.done = False
        self.__reset_robot()
        self.__x_list, self.__y_list = [], []
        self.__number = 0
        self.__reward = 0
        return self.state

    def __reset_robot(self):
        self.__position_robot = np.zeros(2) # The position of the robot [x, y]
        self.__move           = np.zeros(2) # [velocity, angular velocity]
        self.__robot_quat     = np.zeros(4) # Quaternions [Qx, Qy, Qz, Qw]
        
        self.__target_point = [0.6,-0.9]
        # while(1):
        #     self.__target_point = [random.uniform(-AREA_GENERATION,AREA_GENERATION), random.uniform(-AREA_GENERATION,AREA_GENERATION)] # Target point [x, y]
        #     if (self.__target_point[0] != 0 or self.__target_point[1] != 0):
        #         break
            
        self.__old_target_point = self.__target_point
        self.__old_position_robot = 0.0
        
        self.__d_angl_rad = 0.0
        self.__delta_angle = 0.0
        self.__delta_angle_old = 0.0
        
    def step(self, action):
        # Применяем действие к состоянию и получаем новое состояние
        self.__move = action #[v, w]
        print()
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
                                        #  robot_quat_x, robot_quat_y,
                                        #  robot_quat_z, robot_quat_w]
        print(data_vicon)
        self.__position_robot[0], self.__position_robot[1]= data_vicon[0], data_vicon[1]
        self.__robot_quat[2], self.__robot_quat[3] = data_vicon[2], data_vicon[3]
    
        return [self.__position_robot[0] , self.__position_robot[1],
                self.__target_point[0]   , self.__target_point[1]  ,
                self.__move[0]           , self.__move[1]          ,
                self.__robot_quat[2]     , self.__robot_quat[3]    ]
    
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
            self.__old_target_point = [self.__target_point[0], self.__target_point[1]]
            
            self.__old_position_robot = [self.__position_robot[0], self.__position_robot[1]]
            
            self.__delta_angle_old = self.__delta_angle 
           #print(f"reward: {self.__angle_reward()} dist: {self.__dist_reward()}" )
            reward = self.__dist_reward() + self.__angle_reward() 
           # print(f"reward:{reward} angle {self.__angle_reward()} dist: {self.__dist_reward()}" )
            return reward
                       
    def __dist_reward(self):
        dist_new = DistanceBW2points(self.__target_point[0], self.__target_point[1], self.__position_robot[0], self.__position_robot[1]).getDistance()
        dist_old = DistanceBW2points(self.__old_target_point[0], self.__old_target_point[1], self.__old_position_robot[0], self.__old_position_robot[1]).getDistance()
        self.__old_target_point = [self.__target_point[0], self.__target_point[1]]
        self.__old_position_robot = [self.__position_robot[0], self.__position_robot[1]]
        
        if dist_new < dist_old:
            return 1
        else:
            return 0#-17
           
    def __angle_reward(self):
        self.__delta_angle = d_ang(self.__position_robot[0], self.__position_robot[1],
                                   self.__target_point[0], self.__target_point[1],
                                   self.__d_angl_rad
                                   ).get_angle_dev()

        if self.__delta_angle == 0.0: # 17 
            return 1
        # elif abs(self.__delta_angle) < np.pi / 2 and abs(self.__delta_angle) < abs(self.__delta_angle_old):
        #     return -10.83 * self.__delta_angle + 17
        else:
            return 0
              
    def __check_done(self):
        goal = False
        if( - AREA_DEFEAT > self.__position_robot[0] or self.__position_robot[0] > AREA_DEFEAT  or
            - AREA_DEFEAT > self.__position_robot[1] or self.__position_robot[1] > AREA_DEFEAT):
            goal = False
            return True, goal
        
        elif ( self.__target_point[0] + AREA_WIN > self.__position_robot[0] > self.__target_point[0] - AREA_WIN   and
               self.__target_point[1] + AREA_WIN > self.__position_robot[1] > self.__target_point[1] - AREA_WIN):
            goal = True
            self.graf_move()
            return True, goal
        
        else:
            return False, goal
                
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
        plt.savefig('images/plot' + str(self.__number) + '.png', format='png')
        plt.close()