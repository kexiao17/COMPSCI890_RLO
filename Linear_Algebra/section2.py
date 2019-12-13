from IPython.display import display
from rlo import *


class section2(object):
    ##########################################################
    # Second Module - Matrixes transformation
    ##########################################################
    def __init__(self, SCORES, font_size=3):
        self.options = ["Transformation", "Rotation", "Determinant", "Question1", "Question2"]
        self.concept = [self.options[0]]
        self.SCORES = SCORES
        self.font_size = font_size


        DisplayOptions = ["Both", "RED", "BLUE"]
        self.display_option = [DisplayOptions[0]]


        ###################################################
        ## Buttons
        ###################################################
        wOptions = OptionButton(self.concept, options=self.options, update_fcn=self.plot_scene, description="Concept")
        wDisOptions = OptionButton(self.display_option, options=DisplayOptions, update_fcn=self.plot_scene, description="display: ")
        
        ###################################################
        ## Vectors
        ###################################################
        self.vector = Vector([0,5], [0,5])
        wVector = VectorSlider(self.vector, self.plot_scene, value=[5,5],min=-10.0, max=10.0, step=0.1,
                               description="[RED] Vector", continuous_update=True, orientation="horizontal",
                               readout_format=".1f")
        self.matrix = np.identity(2)
        wMatrix = MatrixSlider(self.matrix, self.plot_scene, value=self.matrix, min=-1.0, max=1.0, step=0.01,
                               description="Matrix", continuous_update=True, orientation="horizontal",
                               readout_format=".2f")
        
        ##################################################
        # Angle Slider
        ##################################################
        self.angle = [0]
        wAngle = Slider(self.angle, 0, self.plot_scene, value=10, min=-180, max=180, step=1,
                        description="Angle: ", readout_format=".1f")
        self.question = [0,0]
        wQ1 = PlayButton(self.question, 0, self.plot_scene, value=0, min=0, max=100, step=1,
                            description="Question 1: ")
        wQ2 = PlayButton(self.question, 1, self.plot_scene, value=0, min=0, max=100, step=1,
                            description="Question 2: ")
        
        ##################################################
        # SCORE
        ##################################################
        scorer = ScoreBox(SCORES, 1, self.plot_scene, value=0, min=0, max=10, step=1, description="Score: ")
        
        ##################################################
        # MODULE DESCRIPTION and QUESTION
        ##################################################
        module2_description = f""" <font size="{self.font_size}">
                                This section covers the effect of Transformation, Rotation Matrix on vectors, and try to understand Matrix determinant geometrically.<br>
                                The vector in blue is transformed one, the one in red is the input vector. This section also has 2 question at the end. <br>
                                Try to review the concepts, and answer the question, then enter a self reported score in the Score Box (0 - 10 with 10 being perfect). <br>
                                <b>Instruction</b>: select a concept, then try to play with the slidebars on the right, see how the demo changes, have fun! <br>
                                PS: the Angle slidebar is only responsive to the Rotation option.
                              </font>"""
        module2 = QuestionBox("Description", module2_description)
        
        question1_description = f""" <font size="{self.font_size}">
                                    When select the <b>Question 1</b> in Concept options, try to play with Quesiton 1 Animation button, what do you see? <br>
                                    What is the intrinsic property of a rotation matrix? And how to form a rotation matrix given angle?
                                </font>"""
        question1 = QuestionBox("Question 1", question1_description)
        
        question2_description = f""" <font size="{self.font_size}">
                                    When select the <b>Question 2</b> in Concept options, try to play with Question 2 Animation button, what do you see? <br>
                                    Why is the determinant changing sign (changing color in demo)?<br>
                                    Is there a way to geometrically explain/describe the matrix determinant?
                                </font>"""
        question2 = QuestionBox("Question 2", question2_description)

        matrix_printout = f""" <font size="{self.font_size}">
                              {self.matrix[0,0]:.2f} {self.matrix[0,1]:.2f}<br>
                              {self.matrix[1,0]:.2f} {self.matrix[1,1]:.2f}
                          </font>"""
        matrixDis = QuestionBox("Matrix", matrix_printout)
        
        
        
        items = [widgets.VBox([wOptions.button, wQ1.button, wQ2.button, wDisOptions.button]), 
                 widgets.VBox([wVector.slider, wMatrix.slider, wAngle.slider, scorer.score])]
        whbox = widgets.VBox([module2.text, widgets.HBox(items), question1.text, question2.text])
        self.whbox = whbox
        
    def __call__(self, fig):
        self.init_figure(fig)
        display(self.whbox)

    def init_figure(self,fig):
        ##################################################
        # Init Figure
        ##################################################
        self.fig = fig
        self.fig.clf()
        self.plot_scene()

    def plot_scene(self):
        
        self.fig.clf()
        if self.concept[0] == self.options[0]:
            plot_matrix_transformation(self.fig, self.vector, self.matrix, dp=self.display_option[0])
        elif self.concept[0] == self.options[1]:
            plot_matrix_rotation(self.fig, self.vector, self.matrix, self.angle[0], dp=self.display_option[0])
        elif self.concept[0] == self.options[2]:
            plot_matrix_determinant(self.fig, self.vector, self.matrix, dp=self.display_option[0])
        elif self.concept[0] == self.options[3]:
            plot_matrix_question1(self.fig, self.vector, self.matrix, self.question[0], dp=self.display_option[0])
        elif self.concept[0] == self.options[4]:
            plot_matrix_question2(self.fig, self.vector, self.matrix, self.question[1], dp=self.display_option[0])
    
