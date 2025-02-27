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
import math

from bleak import BleakClient
import asyncio

# Replace this with your BLE device's address
DEVICE_ADDRESS = "XX:XX:XX:XX:XX:XX"

async def connect_to_device(address):
    try:
        # Create a BleakClient object
        async with BleakClient(address) as client:
            print(f"Connected to {address}")

            # Check if the connection is successful
            if client.is_connected:
                print("Device is connected!")

                # Example: Read a specific characteristic
                # Replace with the UUID of the characteristic you want to read
                CHARACTERISTIC_UUID = "00002a00-0000-1000-8000-00805f9b34fb"
                value = await client.read_gatt_char(CHARACTERISTIC_UUID)
                print(f"Value of characteristic {CHARACTERISTIC_UUID}: {value}")

                # Example: Write to a specific characteristic
                # Replace with the UUID of the characteristic you want to write to
                # WRITE_UUID = "00002a01-0000-1000-8000-00805f9b34fb"
                # await client.write_gatt_char(WRITE_UUID, b"your_data_here")

    except Exception as e:
        print(f"An error occurred: {e}")



A = 0.0835
B = 0.153

sigma = math.sin(45 * math.pi / 180)
h = 0.0505
n = 10
c_1 = [A, B]
c_2 = [-A, B]
c_3 = [-A, -B]
c_4 = [A, -B]

k = 360 * n / (2 * math.PI);
start_byte = 50
port = 1

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
    def __init__(self, target_point = [0,0], angle_target = 0):
        self.action_space      = gym.spaces.Box(low=np.array([-A_MAX] * A_SIZE ), high=np.array([A_MAX] * A_SIZE), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([S_MAX  ] * S_SIZE  ), high=np.array([S_MAX ] * S_SIZE ), dtype=np.float32)
        self.__target_point = target_point
        self.__angle_target = angle_target
        self.find_devices()    
        self.target_address = input("Введите адрес устройства для подключения: ")  
        self.reset_env()

    # def find_devices(self):
    #     print("Ищем устройства...")
    #     devices = bluetooth.discover_devices(duration=8, lookup_names=True)
    #     print("Найденные устройства:")
    #     for addr, name in devices:
    #         print(f"{name} - {addr}")
        
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
        self.__robot_quat     = np.zeros(4) # Quaternions [Qx, Qy, Qz, Qw]
        self.__target_point =[random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(S_G_TARG, AREA_GENERATION)]    
        self.__angle_target = np.array([random.uniform(-3.14, 3.14)])[0]
        self.__old_target_point = self.__target_point
        self.__old_position_robot = [0,0]
        
        self.__d_angl_rad = 0.0
        self.__delta_angle = 0.0
        
    def step(self, action):
        # Применяем действие к состоянию и получаем новое состояние
        v1, v2, v3, v4 = self.__calculate_vel(action)
        self.__send_to_robot(v1, v2, v3, v4, self.target_address)
        self.state = self.__get_state()
        # Проверяем, завершен ли эпизод
        self.done, goal = self.__check_done()
        # Вычисляем награду
        self.__reward = self.__new_reward(goal)
        return self.state, self.__reward, self.done, {}
    
    @staticmethod
    def __calculate_vel(action):
        
        v_1 = action[0]* np.cos(action[2]) - action[1] * np.sin(action[2])
        v_2 = action[0]* np.sin(action[2]) - action[1] * np.cos(action[2])
        alpha_1 = [-1 * math.sqrt(2) / 2, math.sqrt(2) / 2]
        alpha_2 = [1 * math.sqrt(2) / 2, math.sqrt(2) / 2]
        alpha_3 = [-1 * math.sqrt(2) / 2, math.sqrt(2) / 2]
        alpha_4 = [1 * math.sqrt(2) / 2, math.sqrt(2) / 2]

        
        vel_1 = (int)(-1 / (sigma * h) * ((v_1 - action[2] * c_1[1]) * alpha_1[0] + (v_2 + action[2] * c_1[0]) * alpha_1[1]) * k);
        vel_2 = (int)(-1 / (sigma * h) * ((v_1 - action[2] * c_2[1]) * alpha_2[0] + (v_2 + action[2] * c_2[0]) * alpha_2[1]) * k);
        vel_3 = (int)(-1 / (sigma * h) * ((v_1- action[2] * c_3[1]) * alpha_3[0] + (v_2 + action[2] * c_3[0]) * alpha_3[1]) * k);
        vel_4 = (int)(-1 / (sigma * h) * ((v_1 - action[2] * c_4[1]) * alpha_4[0] + (v_2 + action[2] * c_4[0]) * alpha_4[1]) * k);
        
        return vel_1, vel_2, vel_3 , vel_4
        
    
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
    def __send_to_robot(vel_1, vel_2, vel_3, vel_4,target_address):
        if (vel_1 > 0):
            dir1 = 1
        else:
            dir1 = 2
            vel_1 = -vel_1
            
        if (vel_2 > 0): 
            vel_2= 1
        else:
            dir2 = 2
            vel_2 = -vel_2
            
        if (vel_3 > 0): 
            dir3 = 1
        else:
            dir3 = 2
            vel_3 = -vel_3
            
        if (vel_4 > 0): 
            dir4 = 1
        else:
            dir4 = 2
            vel_4 = -vel_4
        asyncio.run(connect_to_device(DEVICE_ADDRESS))
        
        
        
        
        
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
        elif ( (self.__target_point[0] + AREA_WIN > self.__position_robot[0] > self.__target_point[0] - AREA_WIN )  and
               (self.__target_point[1] + AREA_WIN > self.__position_robot[1] > self.__target_point[1] - AREA_WIN)):
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