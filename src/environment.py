from src.car import Car
from src.ui import Interface
from src.circuit import Circuit
import numpy as np

def generateCircuitPoints(n_points=16, difficulty=0, circuit_size=(5, 2)):
    n_points = min(25, n_points)
    angles = [-np.pi/4 + 2*np.pi*k/n_points for k in range(3*n_points//4)]
    points = [(circuit_size[0]/2, 0.5), (3*circuit_size[0]/2, 0.5)]
    points += [(circuit_size[0]*(1+np.cos(angle)), circuit_size[1]*(1+np.sin(angle))) for angle in angles]
    for i, angle in zip(range(n_points), angles):
        rd_dist = 0
        if difficulty > 0:
            rd_dist = min(circuit_size) * np.random.vonmises(mu=0, kappa=32/difficulty)/np.pi
        points[i+2] = tuple(np.array(points[i+2]) + rd_dist*np.array([np.cos(angle), np.sin(angle)]))
    return points

class Environment(object):
    NUM_SENSORS = 5

    def __init__(self, circuit, render=False):
        self.circuit = circuit
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)

        # To render the environment
        self.render = render
        if render:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)
        else:
            self.ui = None

        # Build the possible actions of the environment
        self.actions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.actions.append((speed_step, turn_step))

        self.count = 0
        self.progression = 0

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        reward = 0
        if isCrash or hasStopped:
            return -.5
        if self.circuit.laps + self.circuit.progression > self.progression:
            reward += (self.circuit.laps + self.circuit.progression - self.progression)
            self.progression = self.circuit.laps + self.circuit.progression
        # print(reward, self.car.speed/10, self.count/500)
        return reward + self.car.speed/20

    def isEnd(self) -> bool:
        """Is the episode over ?"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        # print(isCrash, hasStopped)
        return isCrash or hasStopped

    def reset(self):
        self.count = 0
        self.progression = 0
        self.car.reset()
        self.circuit.reset()
        return self.current_state

    @property
    def current_state(self):
        result = self.car.distances()
        result.append(self.car.speed)
        return result

    def step(self, i: int, greedy):
        """Takes action i and returns the new state, the reward and if we have
        reached the end"""
        self.count += 1
        self.car.action(*self.actions[i])

        state = self.current_state
        isEnd = self.isEnd()
        reward = self.reward()

        if self.render:
            self.ui.update()

        return state, reward, isEnd

    def mayAddTitle(self, title):
        if self.render:
            self.ui.setTitle(title)
        
    def getNewCircuit(self, n_points=16, difficulty=0):
        print('-------------------\nNew Circuit !\n-------------------')
        self.points = generateCircuitPoints(n_points, difficulty)
        self.circuit = Circuit(self.points)
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)
        self.car.reset()
        self.circuit.reset()
        if self.render:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)
    
    def initRender(self):
        self.render = True
        self.circuit = Circuit(self.circuit.points)
        self.car = Car(self.circuit, num_sensors=self.NUM_SENSORS)
        self.car.reset()
        self.circuit.reset()
        if self.render:
            self.ui = Interface(self.circuit, self.car)
            self.ui.show(block=False)

