from random import randrange

from beamngpy import BeamNGpy, Scenario, Road, Vehicle
from beamngpy.sensors import Camera, Electrics
from matplotlib import pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

import sys
import fileinput

bng = None
vehicle = Vehicle('ego_vehicle', model='etk800', licence='RED', color='blue')
beamng = BeamNGpy('localhost', 64256)
positions = [((70, 0, 0), (0, 0, 275)), ((82, 80, 0), (0, 0, 90)), ((0, 120, 0), (0, 0, 0)), ]

class BeamHandler:

    def __init__(self):

        scenario = Scenario('smallgrid', 'ai_training')
        self.create_scenario(scenario)

        global bng, beamng
        bng = beamng.open(launch=True)

        bng.set_deterministic()  # Set simulator to be deterministic
        bng.set_steps_per_second(60)  # With 60hz temporal resolution

        # Load and start the scenario
        bng.load_scenario(scenario)
        bng.start_scenario()
        # Put simulator in pause awaiting further inputs
        bng.pause()

    def create_scenario(self, scenario):

        road_a = Road('a_asphalt_01_a', rid='road1', looped=True)
        nodes = [
            (0, 0, 0, 10),#
            (33, 0, 0, 10),
            (66, 0, 0, 10),
            (100, 0, 0, 10),#
            (115, 20, 0, 10),
            (130, 40, 0, 10),#
            (115, 60, 0, 10),
            (100, 80, 0, 10),#
            (80, 80, 0, 10),
            (60, 80, 0, 10),#
            (50, 90, 0, 10),
            (40, 100, 0, 10),#
            (40, 120, 0, 10),
            (40, 140, 0, 10),#
            (30, 150, 0, 10),
            (20, 160, 0, 10),#
            (10, 150, 0, 10),
            (0, 140, 0, 10),#
            (0, 120, 0, 10),
            (0, 100, 0, 10),
            (0, 80, 0, 10),
            (0, 60, 0, 10),
            (0, 40, 0, 10),
            (0, 20, 0, 10),
        ]

        road_a.nodes.extend(nodes)
        scenario.add_road(road_a)

        pos = (0, 0.4, 1)
        direction = (0, 3.1415, 0)
        fov = 120
        resolution = (200, 200)
        front_camera = Camera(pos, direction, fov, resolution,
                              colour=True, annotation=True)

        electric = Electrics()

        # Attach them
        global vehicle
        vehicle.attach_sensor('front_cam', front_camera)
        vehicle.attach_sensor('electrics', electric)

        global positions
        x, y = positions[randrange(0, 3)]
        scenario.add_vehicle(vehicle, pos=x, rot=y)

        global beamng
        scenario.make(beamng)

        path_prefab_file = scenario.get_prefab_path()

        for i, line in enumerate(fileinput.input(path_prefab_file, inplace=1)):
            sys.stdout.write(line.replace('overObjects = "0";', 'overObjects = "1";\nannotation="water";'))

    def step(self, action):
        assert vehicle.skt

        actions = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1]
        steering = actions[action]
        sensors = bng.poll_sensors(vehicle)
        speed = sensors['electrics']['values']['wheelspeed']

        throttle = 0.03  # 0.042 = 15 kmh; 0.03 = 10 kmh
        if speed < (6 / 3.6):
            throttle = 0.06

        vehicle.control(throttle=throttle, steering=steering)

        bng.step(30)

        sensors = bng.poll_sensors(vehicle)
        road = beamng.get_road_edges('road1')
        return sensors['front_cam']['annotation'].convert('RGB'), self.calc_reward(road), not self.is_on_track(road)

    def reset(self):
        beamng.restart_scenario()
        global positions
        x, y = positions[randrange(0, 3)]
        beamng.teleport_vehicle(vehicle, x, y)
        sensors = bng.poll_sensors(vehicle)
        return sensors['front_cam']['colour'].convert('RGB')

    def is_on_track(self, road):
        edges = []

        for entry in road:
            left = (entry['left'][0], entry['left'][1])
            edges.append(left)

        to_reverse = []

        for entry in road:
            right = (entry['right'][0], entry['right'][1])
            to_reverse.append(right)

        for entry in to_reverse[::-1]:
            edges.append(entry)

        pos = vehicle.state['pos'].copy()
        del pos[2]
        pos_car = Point(pos)

        road_shape = Polygon(shell=edges)
        #here the road can be drawn by shapely
        #x, y = road_shape.exterior.xy
        #plt.plot(x, y)
        #plt.plot(pos_car.x, pos_car.y)
        #plt.show()
        return road_shape.contains(pos_car)

    def calc_reward(self, road):
        middle_points = []

        for entry in road:
            mid = entry['middle'].copy()
            del mid[2]
            middle_points.append(mid)

        pos = vehicle.state['pos'].copy()
        del pos[2]
        pos_car = Point(pos)

        lane = LineString(middle_points)
        nearest_point = lane.interpolate(lane.project(pos_car))
        distance = pos_car.distance(nearest_point)
        reward = 2.0 - (distance/5.0 + 1)
        if not self.is_on_track(road):
            reward = reward - 2
        return reward


if __name__ == '__main__':
    env = BeamHandler()
