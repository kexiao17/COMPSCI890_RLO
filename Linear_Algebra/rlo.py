import numpy as np
import ipywidgets as widgets
from matplotlib import patches


class Vector(object):
    def __init__(self, x, y, z=None):
        self._x = np.array(x)
        self._y = np.array(y)
        self._z = np.zeros(2) if z is None else z
    
    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, data):
        self._x = data
    
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, data):
        self._y = data
    
    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, data):
        self._z = data

    @property
    def array(self):
        array = np.array([self.x, self.y, self.z]).T
        return array

    @property
    def start(self):
        array = self.array
        return array[0,:]

    @property
    def end(self):
        array = self.array
        return array[1,:]

    @property
    def diff(self):
        array = self.array
        return self.end - self.start
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector(x, y, z)

    def __mul__(self, other):
        p = np.array([self.start, self.start + self.diff * other])
        return Vector.from_array(p)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, idx):
        if idx == 0:
            return self._x[0]
        elif idx == 1:
            return self._y[0]
        elif idx == 2:
            return self._z[0]
        elif idx == 3:
            return self._x[1]
        elif idx == 4:
            return self._y[1]
        elif idx == 5:
            return self._z[1]

    def __setitem__(self, idx, data):
        if idx == 0:
            self._x[0] = data
        elif idx == 1:
            self._y[0] = data
        elif idx == 2:
            self._z[0] = data
        elif idx == 3:
            self._x[1] = data
        elif idx == 4:
            self._y[1] = data
        elif idx == 5:
            self._z[1] = data

    def xlims(self, threshold, margin=0.5):
        threshold = np.abs(threshold) + margin

        xlim0 = min(self.x.min() - margin, -threshold)
        xlim1 = max(self.x.max() + margin, threshold)

        return (xlim0, xlim1)

    def ylims(self, threshold, margin=0.5):
        threshold = np.abs(threshold) + margin

        ylim0 = min(self.y.min() - margin, -threshold)
        ylim1 = max(self.y.max() + margin, threshold)

        return (ylim0, ylim1)

    def zlims(self, threshold, margin=0.5):
        threshold = np.abs(threshold) + margin

        zlim0 = min(self.z.min() - margin, -threshold)
        zlim1 = max(self.z.max() + margin, threshold)

        return (zlim0, zlim1)

    def lims(self, threshold, margin=0.5):
        xlim0, xlim1 = self.xlims(threshold=threshold, margin=margin)
        ylim0, ylim1 = self.ylims(threshold=threshold, margin=margin)
        zlim0, zlim1 = self.zlims(threshold=threshold, margin=margin)

        return ((xlim0, xlim1), (ylim0, ylim1), (zlim0, zlim1))


    @classmethod
    def from_array(Kls, array):
        if array.ndim == 1:
            array = np.array([np.zeros_like(array), array])
        if array.shape[1] == 2:
            array = np.concatenate([array.T, np.zeros((1,2))]).T

        x = array[:,0]
        y = array[:,1]
        z = array[:,2]

        return Kls(x, y, z)

    def project(self, vector):
        v = np.dot(self.diff, vector.diff) / np.dot(vector.diff, vector.diff) * vector
        dotted = Vector.from_array(np.array([v.end, self.end]))
        return v, dotted

    def translate(self, vector):
        array = np.array([self.start + vector.diff, self.end + vector.diff])
        return Vector.from_array(array)

    def transform(self, matrix):
        vectorT = np.dot(matrix, self.diff)
        arrayT = np.array([self.start, self.start+vectorT])
        return Vector.from_array(arrayT)

    def decompose(self, vectorX, vectorY):
        v = self.diff
        vx = vectorX.diff / np.linalg.norm(vectorX.diff)
        vy = vectorY.diff / np.linalg.norm(vectorY.diff)

        a1 = (v[1] * vy[0] - v[0] * vy[1]) / (vx[1] * vy[0] - vx[0] * vy[1])
        a2 = (v[1] * vx[0] - v[0] * vx[1]) / (vy[1] * vx[0] - vy[0] * vx[1])

        vx = Vector.from_array(np.array([self.start, self.start + a1 * vx]))
        vy = Vector.from_array(np.array([self.start, self.start + a2 * vy]))

        dottedX = Vector.from_array(np.array([vx.end, self.end]))
        dottedY = Vector.from_array(np.array([vy.end, self.end]))

        return vx, vy, dottedX, dottedY

class Slider(object):
    def __init__(self, box, idx, update_fcn, value, min, max, step, description,
                 continuous_update=True, orientation="horizontal", readout_format=".2f"):

        self.box = box
        self.idx = idx
        self.update_fcn = update_fcn

        self.slider = widgets.FloatSlider(
            value = value,
            min = min,
            max = max,
            step = step,
            description = description,
            disable = False,
            continous_update = continuous_update,
            orientation=orientation,
            readout=True,
            readout_format=readout_format,
        )
        self.slider.observe(self.on_slider_change, names='value')

    def on_slider_change(self, change):
        self.box[self.idx] = float(change.new)
        self.update_fcn()


class VectorSlider(object):
    def __init__(self, box, update_fcn, value, min, max, step, description,
                 continuous_update=True, orientation="horizontal", readout_format=".2f"):

        self.sliderX = Slider(box, 3, update_fcn, value[0], min, max, step, f"{description}.X",
                              continuous_update=continuous_update, orientation=orientation,
                              readout_format=readout_format)
        self.sliderY = Slider(box, 4, update_fcn, value[1], min, max, step, f"{description}.Y",
                              continuous_update=continuous_update, orientation=orientation,
                              readout_format=readout_format)

        self.slider = widgets.VBox([self.sliderX.slider, self.sliderY.slider])

class MatrixSlider(object):
    def __init__(self, box, update_fcn, value, min, max, step, description,
                 continuous_update=True, orientation="horizontal", readout_format=".2f"):

        idxes = [(0,0),(0,1),(1,0),(1,1)]
        self.sliders = []
        for idx in idxes:
            slider = Slider(box, idx, update_fcn, value[idx], min, max, step, f"{description} {idx}",
                            continuous_update=continuous_update, orientation=orientation,
                            readout_format=readout_format)
            self.sliders.append(slider)

        self.slider = widgets.VBox([x.slider for x in self.sliders])

class OptionButton(object):
    def __init__(self, box, options, update_fcn, description="Option"):

        self.box = box
        self.update_fcn = update_fcn

        self.button = widgets.RadioButtons(
            options=options,
            description = description,
            disable = False,
        )
        self.button.observe(self.on_button_change, names='value')

    def on_button_change(self, change):
        self.box[0] = str(change.new)
        self.update_fcn()

class ScoreBox(object):
    def __init__(self, box, idx, update_fcn, value, min, max, step, description):

        self.box = box
        self.idx = idx
        self.update_fcn = update_fcn

        self.score = widgets.BoundedIntText(
            value = value,
            min = min,
            max = max,
            step = step,
            description = description,
            disable = False,
        )
        self.score.observe(self.on_score_change, names='value')

    def on_score_change(self, change):
        self.box[self.idx] = int(change.new)
        self.update_fcn()

class PlayButton(object):
    def __init__(self, box, idx, update_fcn, value, min, max, step, description):

        self.box = box
        self.idx = idx
        self.update_fcn = update_fcn

        self.caption = widgets.Label(value=description)
        self.play = widgets.Play(
            value = value,
            min = min,
            max = max,
            step = step,
            description = description,
            disable = False
        )

        self.play.observe(self.on_play_change, names='value')
        self.button = widgets.HBox([self.caption, self.play])

    def on_play_change(self, change):
        self.box[self.idx] = float(change.new)
        self.update_fcn()




def plot_vector(fig, vector, color='black', arrow_size=0.5, text=True, **kwargs):
    ax = fig.add_subplot(1,1,1)

    head_width = arrow_size
    head_length = arrow_size

    xlims = vector.xlims(threshold=10., margin=0.5)
    ylims = vector.ylims(threshold=10., margin=0.5)


    ax.arrow(vector.x[0], vector.y[0],
             vector.diff[0], vector.diff[1],
             head_width=head_width,
             head_length=head_length,
             fc=color, ec=color,
             **kwargs)
    
    if isinstance(text, str):
        alpha = 1.2
        end_x = alpha * vector.x[1] + (1-alpha)*vector.x[0]
        end_y = alpha * vector.y[1] + (1-alpha)*vector.y[0]
        ax.text(end_x, end_y, 
                text, color=color,
                size=10, ha='center', va='center')
    elif text:
        alpha = 1.1
        end_x = alpha * vector.x[1] + (1-alpha)*vector.x[0]
        end_y = alpha * vector.y[1] + (1-alpha)*vector.y[0]
        ax.text(end_x, end_y, 
                f"({vector.x[1]:.2f},{vector.y[1]:.2f})",
                color=color,
                size=10, ha='center', va='center')
        
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.grid(True)
    ax.set_xlabel("Vector.X")
    ax.set_ylabel("Vector.Y")

def plot_two_vectors(fig, vector1, vector2):
    plot_vector(fig, vector1, 'red')
    plot_vector(fig, vector2, 'blue')

    ax = fig.add_subplot(1,1,1)
    title = "2D Vectors"
    ax.set_title(title)

def plot_vector_addition(fig, vector1, vector2):
    vector1T = vector1.translate(vector2)
    vector2T = vector2.translate(vector1)
    vector1a2 = vector1 + vector2

    plot_vector(fig, vector1, 'red')
    plot_vector(fig, vector2, 'blue')
    plot_vector(fig, vector1T, 'red', 0.05, False, linestyle=(5, (3,3)))
    plot_vector(fig, vector2T, 'blue', 0.05, False, linestyle=(5, (3,3)))
    plot_vector(fig, vector1a2, 'green')

    ax = fig.add_subplot(1,1,1)
    title = f"[RED]V1 {vector1.diff[:2]} + [BLUE]V2 {vector2.diff[:2]} = [GRENN]V3 {vector1a2.diff[:2]}"
    ax.set_title(title)

def get_patch(vector1, vector2, color='green', alpha=None):
    vector1a2 = vector1 + vector2
    coords = [vector1.start[:2], vector1.end[:2], vector1a2.end[:2], vector2.end[:2]]
    coords = np.array(coords)
    p = patches.Polygon(coords, True, color=color, alpha=alpha)

    return p

def plot_vector_cross_product(fig, vector1, vector2):
    ax = fig.add_subplot(1,1,1)

    vector1T = vector1.translate(vector2)
    vector2T = vector2.translate(vector1)
    vector1a2 = vector1 + vector2

    plot_vector(fig, vector1, 'red')
    plot_vector(fig, vector2, 'blue')
    plot_vector(fig, vector1T, 'red', 0.05, False, linestyle=(5, (3,3)))
    plot_vector(fig, vector2T, 'blue', 0.05, False, linestyle=(5, (3,3)))

    p = get_patch(vector1, vector2, color='purple')
    ax.add_patch(p)

    area = np.cross(vector1.diff, vector2.diff)[-1]

    title = f"[RED]V1 {vector1.diff[:2]} x [BLUE]V2 {vector2.diff[:2]} = [Shaded Area] {area:.2f}"
    ax.set_title(title)

def plot_vector_projection(fig, vector1, vector2):
    vector1P, vector1D = vector1.project(vector2)
    vector2P, vector2D = vector2.project(vector1)

    plot_vector(fig, vector1, 'red')
    plot_vector(fig, vector2, 'blue')
    plot_vector(fig, vector1P, 'red', 0.4, True, linestyle=(5, (3,3)))
    plot_vector(fig, vector2P, 'blue', 0.4, True, linestyle=(5, (3,3)))
    plot_vector(fig, vector1D, 'red', 0.05, False, linestyle=(5, (3,3)))
    plot_vector(fig, vector2D, 'blue', 0.05, False, linestyle=(5, (3,3)))

    ax = fig.add_subplot(1,1,1)
    title = "Vector Projection"
    ax.set_title(title)

def plot_vector_question1(fig, alpha, vector1, vector2):
    vector3 = alpha * vector1 + (1-alpha) * vector2
    dotted = Vector.from_array(np.array([vector1.end, vector2.end]))

    plot_vector(fig, vector1, 'red')
    plot_vector(fig, vector2, 'blue')
    plot_vector(fig, dotted, 'orange', 0.05, False, linestyle=(5, (3,3)))
    plot_vector(fig, vector3, 'orange')

    ax = fig.add_subplot(1,1,1)
    title = "How to prove alpha * V1[RED] + (1-alpha) * V2[BLUE] = V3[ORANGE]?"
    ax.set_title(title)

def plot_vector_question1_animation(fig, q1, vector1, vector2):
    alpha = q1 / 100. * 2. - .5
    plot_vector_question1(fig, alpha, vector1, vector2)
    


def prep_matrix(matrix):
    if matrix.shape == (3,3):
        return matrix

    Matrix = np.identity(3)
    s1, s2 = matrix.shape
    Matrix[:s1, :s2] = matrix

    return Matrix

def get_axes(matrix):
    I = np.identity(3)
    Ix, Iy = Vector.from_array(I[:,0]), Vector.from_array(I[:,1])
    Vx, Vy = Ix.transform(matrix), Iy.transform(matrix)
    Vx = (Vx * 20.).translate(Vx * -10)
    Vy = (Vy * 20.).translate(Vy * -10)

    return Vx, Vy


def plot_vector_decompose(fig, vector, axes, color="black", patch=False, alpha=None, color2=None, det=1.0):
    axX, axY = axes
    vx, vy, dottedX, dottedY = vector.decompose(axX, axY)

    plot_vector(fig, axX, color, 0.3, "X", linestyle=(5, (3,3)))
    plot_vector(fig, axY, color, 0.3, "Y", linestyle=(5, (3,3)))
    plot_vector(fig, dottedX, color, 0.03, False, linestyle=(5, (3,3)))
    plot_vector(fig, dottedY, color, 0.03, False, linestyle=(5, (3,3)))

    if patch:
        color2 = color if color2 is None else color2
        color = color if det > 0 else color2
        p = get_patch(vx, vy, color=color, alpha=alpha)
        ax = fig.add_subplot(1,1,1)
        ax.add_patch(p)
        



def plot_matrix_transformation(fig, vector, matrix, dp=0):
    matrix = prep_matrix(matrix)
    I = np.identity(3)

    axesI = get_axes(I)
    axesT = get_axes(matrix)
    vectorT = vector.transform(matrix)

    plot_vector(fig, vector, 'red', 0.5, True)
    plot_vector(fig, vectorT, 'blue', 0.5, True)

    if dp == "Both":
        plot_vector_decompose(fig, vector, axesI, 'red')
        plot_vector_decompose(fig, vectorT, axesT, 'blue')
    elif dp == "RED":
        plot_vector_decompose(fig, vector, axesI, 'red')
    elif dp == "BLUE":
        plot_vector_decompose(fig, vectorT, axesT, 'blue')


    ax = fig.add_subplot(1,1,1)
    title = "Matrix Transformation [RED] -> [BLUE]"
    ax.set_title(title)



def plot_matrix_determinant(fig, vector, matrix, dp=0):
    matrix = prep_matrix(matrix)
    I = np.identity(3)

    axesI = get_axes(I)
    axesT = get_axes(matrix)
    vectorT = vector.transform(matrix)

    plot_vector(fig, vector, 'red', 0.5, True)
    plot_vector(fig, vectorT, 'blue', 0.5, True)

    det = np.linalg.det(matrix)

    if dp == "Both":
        plot_vector_decompose(fig, vector, axesI, 'red', patch=True, alpha=1.0)
        plot_vector_decompose(fig, vectorT, axesT, 'blue', patch=True, alpha=0.5, color2="cyan", det=det)
    elif dp == "RED":
        plot_vector_decompose(fig, vector, axesI, 'red', patch=True, alpha=1.0)
    elif dp == "BLUE":
        plot_vector_decompose(fig, vectorT, axesT, 'blue', patch=True, alpha=0.5, color2="cyan", det=det)

    ax = fig.add_subplot(1,1,1)
    title = f"Matrix Determinant = Area[BLUE]/Area[RED] = {det:.2f}"
    ax.set_title(title)



def plot_matrix_rotation(fig, vector, matrix, angle, dp=0):
    matrix[0,0] = np.cos(angle / 180. * np.pi)
    matrix[0,1] = - np.sin(angle / 180. * np.pi)
    matrix[1,0] = - matrix[0,1]
    matrix[1,1] = matrix[0,0]
    plot_matrix_transformation(fig, vector, matrix, dp=dp)

    ax = fig.add_subplot(1,1,1)
    title = f"Matrix Rotation (Angle: {angle}) [RED] -> [BLUE]"
    ax.set_title(title)


def plot_matrix_question1(fig, vector, matrix, q1, dp=0):
    angle = q1 / 100 * 360
    plot_matrix_rotation(fig, vector, matrix, angle, dp=dp)

    ax = fig.add_subplot(1,1,1)
    title = f"Q1: How to form a rotation matrix ?"
    ax.set_title(title)

def plot_matrix_question2(fig, vector, matrix, q2, dp=0):
    matrix[1,1] = (q2 / 100. * 2)*(-1.) + 1
    plot_matrix_determinant(fig, vector, matrix, dp=dp)

    det = np.linalg.det(matrix)
    ax = fig.add_subplot(1,1,1)
    title = f"How does the matrix change and \nwhy is the determinant ({det:.2f}) flipping sign (changing color)?"
    ax.set_title(title)






