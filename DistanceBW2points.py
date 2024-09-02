from numpy import sqrt


class DistanceBW2points:

    def __init__(self, first_x, first_y, second_x, second_y):
        self.x1, self.y1, self.x2, self.y2 = first_x, first_y, second_x, second_y

    def getDistance(self):
        x = pow(self.x2 - self.x1, 2)
        y = pow(self.y2 - self.y1, 2)
        return sqrt(x + y)


def main():
    pass


if __name__ == "__main__":
    main()