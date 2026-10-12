import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

HOUSE_COLORS = {
    'Gryffindor': '#ae0001',
    'Hufflepuff': '#e3a000',
    'Ravenclaw': '#1f4e9c',
    'Slytherin': '#2a7a3f',
}


class TrainingPlot:
    """
        Live view of a LogisticModel training: loss of each one-vs-all classifier
        and, when trained on exactly two features, the decision regions.
        Meant to be passed as the on_progress callback of LogisticModel.train.
    """

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    grid_size = 200

    def __init__(self, features, iterations, live=True) -> None:
        self.features = features
        self.iterations = iterations
        self.live = live
        self.show_regions = len(features) == 2
        self.history = {'it': [], **{house: [] for house in TrainingPlot.houses}}

        if self.show_regions:
            self.figure, (self.loss_ax, self.region_ax) = plt.subplots(1, 2, figsize=(13, 6))
        else:
            self.figure, self.loss_ax = plt.subplots(figsize=(7, 6))
            self.region_ax = None

        self.loss_lines = {}
        for house in TrainingPlot.houses:
            self.loss_lines[house], = self.loss_ax.plot([], [], color=HOUSE_COLORS[house], label=house, linewidth=2)
        self.loss_ax.set_xlim(0, iterations)
        self.loss_ax.set_xlabel('Iteration')
        self.loss_ax.set_ylabel('Cross-entropy loss')
        self.loss_ax.set_title('Loss per one-vs-all classifier')
        self.loss_ax.legend(loc='upper right')
        self.loss_ax.grid(alpha=0.3)

        self.region_image = None
        self.title = self.figure.suptitle('')

        if self.live:
            plt.ion()
            plt.show()

    def _init_regions(self, notes, binary_dict):
        x, y = notes[0], notes[1]
        margin = 0.05
        self.extent = (x.min() - margin, x.max() + margin, y.min() - margin, y.max() + margin)
        gx = np.linspace(self.extent[0], self.extent[1], TrainingPlot.grid_size)
        gy = np.linspace(self.extent[2], self.extent[3], TrainingPlot.grid_size)
        self.grid_x, self.grid_y = np.meshgrid(gx, gy)
        self.house_rgb = np.array([to_rgb(HOUSE_COLORS[house]) for house in TrainingPlot.houses])

        for house in TrainingPlot.houses:
            mask = np.array(binary_dict[house]) == 1.0
            self.region_ax.scatter(x[mask], y[mask], s=8, color=HOUSE_COLORS[house],
                                   edgecolors='white', linewidths=0.3, label=house)

        self.region_image = self.region_ax.imshow(np.zeros((TrainingPlot.grid_size, TrainingPlot.grid_size, 4)),
                                                  extent=self.extent, origin='lower', aspect='auto', zorder=0)
        self.region_ax.set_xlabel(f'{self.features[0]} (normalized)')
        self.region_ax.set_ylabel(f'{self.features[1]} (normalized)')
        self.region_ax.set_title('Decision regions')

    def _update_regions(self, intercept, weight):
        scores = np.array([
            1.0 / (1.0 + np.exp(-(intercept[house] + weight[house][0] * self.grid_x + weight[house][1] * self.grid_y)))
            for house in TrainingPlot.houses
        ])
        winner = np.argmax(scores, axis=0)
        # Share of the winning score: 0.25 when the model can't tell houses apart, 1.0 when it is sure
        confidence = scores.max(axis=0) / scores.sum(axis=0)
        rgba = np.zeros((TrainingPlot.grid_size, TrainingPlot.grid_size, 4))
        rgba[..., :3] = self.house_rgb[winner]
        rgba[..., 3] = 0.45 * (confidence - 0.25) / 0.75
        self.region_image.set_data(rgba)

    def update(self, it, intercept, weight, cost_dict, notes, binary_dict):
        self.history['it'].append(it)
        for house in TrainingPlot.houses:
            self.history[house].append(cost_dict[house])
            self.loss_lines[house].set_data(self.history['it'], self.history[house])
        self.loss_ax.set_ylim(0, max(max(self.history[house]) for house in TrainingPlot.houses) * 1.05)

        if self.show_regions:
            if self.region_image is None:
                self._init_regions(notes, binary_dict)
            self._update_regions(intercept, weight)

        self.title.set_text(f'Iteration {it} / {self.iterations}')

        if self.live:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)

    def keep_open(self):
        if self.live:
            plt.ioff()
            plt.show()
