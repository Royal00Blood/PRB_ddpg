from numpy import arctan2, array, pi


class DeviationAngle:

    def __init__(self, robot_x, robot_y, goal_x, goal_y, angle_robot):
        self.__v_rob = array([robot_x, robot_y])
        self.__v_goal = array([goal_x, goal_y])
        self.__angle_r = angle_robot

    def get_angle_dev(self):
        angle_t = arctan2(self.__v_goal[1] - self.__v_rob[1], self.__v_goal[0] - self.__v_rob[0])
        return abs(angle_t - self.__angle_r)


def main():
    vector = DeviationAngle(1, 2, 4, 5, pi / 4)
    print(vector.get_angle_dev())


if __name__ == "__main__":
    main()