import shapely.geometry as geom
from descartes import PolygonPatch
import numpy as np

class Circuit(object):

    def __init__(self, points, width=0.3, num_checkpoints=100):
        self.points = points
        if self.points[0] != self.points[-1]:
            self.points.append(points[0])

        # Compute circuit's geometry
        self.line = geom.LineString(self.points)
        self.width = width
        self.circuit = self.line.buffer(self.width, cap_style=1)

        # For numerical stabilities when checking if something is inside the
        # circuit.
        self.dilated = self.line.buffer(self.width * 1.01, cap_style=1)

        # Where the start line is
        self.defineStart()

        # Define the checkpoints
        self.makeCheckpoints(n=num_checkpoints)

    def defineStart(self):
        """The start line is in the middle of the longest horizontal segment."""
        last = geom.Point(*self.line.coords[0])
        self.start = last
        maxDistance = 0
        for x, y in self.line.coords[1:]:
            curr = geom.Point((x, y))
            if curr.distance(last) > maxDistance and curr.y == last.y:
                maxDistance = curr.distance(last)
                self.start = geom.Point(
                    (0.5 * (x + last.x)), 0.5 * (y + last.y))
            last = curr

        self.start_line = geom.LineString([
            (self.start.x, self.start.y - self.width),
            (self.start.x, self.start.y + self.width)])

    def makeCheckpoints(self, n):
        step_ext = self.circuit.exterior.length / n
        step_int = self.circuit.interiors[0].length / n
        self.checklines = []
        for i in range(n):
            self.checklines.append(geom.LineString([
                self.circuit.exterior.interpolate(step_ext * (n - i)),
                self.circuit.interiors[0].interpolate(step_int * i)],
            ))
        self.reset()

    def reset(self):
        self.checkpoints = [False for i in self.checklines]
        self.laps = 0
        self.progression = 0

    def updateCheckpoints(self, obj):
        if not all(self.checkpoints):
            for idx, line in enumerate(self.checklines):
                if line.intersects(obj):
                    self.checkpoints[idx] = True

        if all(self.checkpoints):
            if self.start_line.intersects(obj):
                self.checkpoints = [False for i in range(len(self.checklines))]
                self.laps += 1

        done = len(list(filter(None, self.checkpoints)))
        self.progression = done / len(self.checkpoints)

    def debug(self):
        return "laps {}: {:.0f}%".format(self.laps, self.progression * 100)

    def __contains__(self, shape):
        return self.dilated.contains(shape)

    def plot(self, ax, color='gray', skeleton=True):
        if skeleton:
            ax.plot(
                self.line.xy[0], self.line.xy[1],
                color='white', linewidth=3, solid_capstyle='round', zorder=3,
                linestyle='--')

        ax.plot(
            self.start_line.xy[0], self.start_line.xy[1],
            color='black', linewidth=3, linestyle='-', zorder=3)

        patch = PolygonPatch(
            self.circuit, fc=color, ec='black', alpha=0.5, zorder=2)
        ax.add_patch(patch)

        bounds = self.circuit.bounds
        offset_x = (bounds[2] - bounds[0]) * 0.1
        offset_y = (bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - offset_x, bounds[2] + offset_x)
        ax.set_ylim(bounds[1] - offset_y, bounds[3] + offset_y)
        ax.set_aspect(1)


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