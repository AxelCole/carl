from src.car import Car
from src.circuit import Circuit
from src.ui import Interface


class Environment(object):
    def __init__(self, num_sensors=5, render=False, crash_value=0):

        # The geometry of the circuit and the position of the car.
        coords = [(0, 0), (0.5, 1), (0, 2), (2, 2), (3, 1), (6, 2), (6, 0)]
        self.circuit = Circuit(coords, width=0.3)

        self.x0 = 3.0
        self.y0 = 0.0
        self.w_car = 0.2
        self.h_car = 0.4
        self.theta = 0.0
        self.speed0 = 0.0
        self.num_sensors = num_sensors
        self.car = Car(
            self.x0, self.y0, self.w_car, self.h_car, self.theta, self.speed0,
            circuit=self.circuit, num_sensors=self.num_sensors)

        # To render the environment
        self.ui = Interface(self.circuit, self.car)
        self.render = render
        if render:
            self.ui.show(block=False)

        # Build the possible actions of the environment
        self.actions = []
        for turn_step in range(-2, 3, 1):
            for speed_step in range(-1, 2, 1):
                self.actions.append((speed_step, turn_step))

        # Used for the rewards
        self.crash_value = crash_value
        self.count = 0

    def reward(self) -> float:
        """Computes the reward at the present moment"""
        isCrash = self.car.car not in self.circuit
        unit = self.car.speed / self.car.SPEED_UNIT
        return unit ** 1.1 + int(isCrash) * self.crash_value

    def isEnd(self) -> bool:
        """Is the episode over ?"""
        isCrash = self.car.car not in self.circuit
        hasStopped = self.car.speed < self.car.SPEED_UNIT
        return isCrash or hasStopped

    def reset(self):
        self.count = 0
        self.car.reset(
            self.x0, self.y0, self.w_car, self.h_car, self.theta, self.speed0)
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
