import matplotlib.pyplot as plt
import numpy as np
import copy
import math

class FeatExt():
    def __init__(self, image, env):
        self.env = env
        self.layer_list = []
        self.layer_list.append([7, np.arange(0, 0.8, 0.1)*(np.pi)])
        self.layer_list.append([12, np.arange(0, 0.7, 0.1)*(np.pi)])
        self.layer_list.append([20, np.arange(0, 0.5, 0.075)*(np.pi)])
        self.layer_list.append([30, np.arange(0, 0.35, 0.05)*(np.pi)])
        self.layer_list.append([40, np.arange(0, 0.3, 0.04)*(np.pi)])
        self.layer_list.append([50, np.arange(0, 0.25, 0.035)*(np.pi)])
        self.layer_list.append([60, np.arange(0, 0.2, 0.035)*(np.pi)])
        self.layer_list.append([70, np.arange(0, 0.2, 0.035)*(np.pi)])
        self.sensor_filter = np.zeros_like(image)
        self.nearest_filter = np.zeros_like(image)
        self.feature_count = 0
        for layer in self.layer_list:
            r, base = layer[0], layer[1]
            x_axs, y_axs = np.cos(base)*r, np.sin(base)*r
            for i in range(len(x_axs)):
                self.sensor_filter[-int(x_axs[i])+70, int(y_axs[i])+48] = 1
                self.sensor_filter[-int(x_axs[i])+70, -int(y_axs[i])+47] = 1
                self.feature_count += 2
        self.nearest_layer = self.layer_list[0]
        r, base = self.nearest_layer[0], self.nearest_layer[1]
        x_axs, y_axs = np.cos(base)*r, np.sin(base)*r
        for i in range(len(x_axs)):
            self.nearest_filter[-int(x_axs[i])+70, int(y_axs[i])+48] = 1
            self.nearest_filter[-int(x_axs[i])+70, -int(y_axs[i])+47] = 1

        self.num_of_lidar_points = 7

        self.no_move_counter = -50
        self.off_road_counter = 0
        self.tolerance = 1

    @staticmethod
    def __set_point_value(s_, point, list_=False):
        if not list_:
            s_[point[1], point[0], :] = 255
            return s_
        else:
            for a_point in point:
                s_[a_point[1], a_point[0], :] = 255
            return s_

    @staticmethod
    def __calculate_circle_points(radius, center_point, num_of_points=7):
        assert num_of_points % 2 == 1

        initial_upright_point = [center_point[0], center_point[1] - radius]
        deg_interval = math.pi / (num_of_points - 1)
        rotation_values_rad = []
        for i in range(num_of_points):
            rotation_values_rad.append((i * deg_interval) - math.pi / 2)

        rotated_points = []
        # translate point back to origin
        initial_upright_point[0] -= center_point[0]
        initial_upright_point[1] -= center_point[1]

        for rotation_value_rad in rotation_values_rad:
            # rotate point
            c = math.cos(rotation_value_rad)
            s = math.sin(rotation_value_rad)

            x1 = initial_upright_point[0] * c - initial_upright_point[1] * s
            y1 = initial_upright_point[0] * s + initial_upright_point[1] * c
            # translate point back
            rotated_points.append([round(x1 + center_point[0]), round(y1 + center_point[1])])

        return rotated_points

    def __get_car_kinematics(self):
        self.car_position_x = self.env.car.hull.position.x
        self.car_position_y = self.env.car.hull.position.y
        self.car_pos_angle = self.env.car.hull.angle
        # print(self.car_pos_angle)
        self.car_angularDamping = self.env.car.hull.angularDamping
        self.car_angularVelocity = self.env.car.hull.angularVelocity
        self.car_velocity_x = self.env.car.hull.linearVelocity.x
        self.car_velocity_y = self.env.car.hull.linearVelocity.y
        self.car_linearDamping = self.env.car.hull.linearDamping

    def __get_lidar_features_img(self, s_):
        self.__get_car_kinematics()
        y_ = 66
        x1_ = 47
        x2_ = 48
        r1 = 10
        car_angle = self.car_pos_angle
        car_angle += math.pi / 2
        # print(car_angle*180/math.pi)

        # point1_upright_left = [x1_, y_ - r1]
        # point1_upright_right = [x2_, y_ - r1]
        # self.__set_point_value(s_, point1_upright_left)
        # self.__set_point_value(s_, point1_upright_right)
        rotated_circle_1 = self.__calculate_circle_points(radius=r1,
                                                          center_point=[x1_, y_],
                                                          num_of_points=self.num_of_lidar_points)
        # self.__set_point_value(s_, rotated_circle_1, list=True)
        distance_list_1 = []
        for a_point in rotated_circle_1:
            x_rate = ((a_point[0] - x1_) / r1) * 2
            y_rate = ((a_point[1] - y_) / r1) * 2

            found_lane_flag = False
            start_point = [x1_, y_]
            next_point = copy.deepcopy(start_point)
            while not found_lane_flag:
                next_point[0] = round(next_point[0] + x_rate)
                next_point[1] = round(next_point[1] + y_rate)
                if next_point[0] > 95 or next_point[1] > 95:
                    break
                pixel_value = s_[next_point[1], next_point[0]]
                if not (((pixel_value > [30, 30, 30]).all()) and (pixel_value < [150, 150, 150]).all()):
                    break

                self.__set_point_value(s_, next_point)

            distance_list_1.append(math.sqrt((next_point[0] - x1_) ** 2 + (next_point[1] - y_) ** 2)/60)

        s_[y_, x1_, :] = 255
        s_[y_, x2_, :] = 255
        return distance_list_1, s_

    def feature_extractor(self, im):
        sensor_im = im * self.sensor_filter
        feats = sensor_im[sensor_im[:, :, 0] > 0]
        sensor_out = np.asarray([feats[:, 1] < 180], dtype=np.float)[0]

        nearest_im = im * self.nearest_filter
        near_feats = nearest_im[nearest_im[:, :, 1] > 0]
        off_road = float(np.sum(np.asarray([near_feats[:, 1] > 200], dtype=np.float)) >= 14)

        #plt.imshow((im * (1 - self.sensor_filter)))
        #plt.show()

        return sensor_out, off_road

    def feat2state(self, s, env):

        state_sens, state_or = self.feature_extractor(s)
        state_lidar, _ = self.__get_lidar_features_img(s)


        state_linvel, state_angvel = np.asarray(env.env.car.hull.linearVelocity)/60, np.asarray(env.env.car.hull.angularVelocity)
        state_speed = np.asarray(np.linalg.norm(state_linvel))
        state_no_move_counter, state_or_counter = np.asarray(self.no_move_counter/self.tolerance)[np.newaxis, ...], np.asarray(self.off_road_counter/self.tolerance)[np.newaxis, ...]
        state_lindamp, state_angdamp = np.asarray(env.env.car.hull.linearDamping), np.asarray(env.env.car.hull.angularDamping)
        state_carangle = np.asarray(env.env.car.hull.angle)

        state = np.concatenate([state_sens, state_lidar, state_carangle[np.newaxis, ...], state_lindamp[np.newaxis, ...], state_angdamp[np.newaxis, ...], state_speed[np.newaxis, ...], state_no_move_counter, state_or_counter, np.asarray(state_or)[np.newaxis, ...], state_linvel, state_angvel[np.newaxis, ...]])
        if state.shape[0] < 131:
            #plt.imshow(s)
            #plt.show()
            print(s.shape)
            print(state_sens.shape)
            print(state_angvel.shape)
            print(state_linvel.shape)
            print(state_or)
        kill_signal = 0
        if state_or > 0:
            self.off_road_counter += 1
        else:
            self.off_road_counter = 0

        if state_speed*60 < 2:
            self.no_move_counter += 1
        else:
            if self.no_move_counter < 0:
                pass
            else:
                self.no_move_counter = -15

        if self.off_road_counter >= self.tolerance or self.no_move_counter >= self.tolerance:
            if self.no_move_counter == self.tolerance:
                print("KILLED DUE TO --- NO MOVE ---")
            if self.off_road_counter == self.tolerance:
                print("KILLED DUE TO --- OFF ROAD ---")
            self.no_move_counter = -60
            kill_signal = 1

        return state, kill_signal

    def networkAct2envAct(self, act):
        env_act = np.zeros(3) # Steer, Gas, Brake
        env_act[0] = act[0]
        borderAccDoNot, borderDoNotBrake = -0.025, -0.075
        if act[1] > borderAccDoNot:
            env_act[1] = (act[1]-borderAccDoNot) / (1-borderAccDoNot) * 0.7
        elif act[1] > borderDoNotBrake:
            pass
        else:
            env_act[2] = (-act[1]-borderDoNotBrake) / (1+borderDoNotBrake)
        return env_act
