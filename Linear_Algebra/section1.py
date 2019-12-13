from IPython.display import display
from rlo import *

class section1(object):
    ##########################################################
    # First Module - Vectors, matrixes
    ##########################################################
    def __init__(self, SCORES, font_size=3):
        self.options = ["2D Vectors", "Vector Addition", "Vector Cross Product", "Vector Projection", 
                        "Question 1", "Question 1 Animation"]
        self.concept = [self.options[0]]
        self.font_size = font_size

        ###################################################
        ## Buttons
        ###################################################
        wOptions = OptionButton(self.concept, options=self.options, update_fcn=self.plot_scene, description="Concept")
    
        ###################################################
        ## Vectors
        ###################################################
        self.vector1 = Vector([0,5], [0,5])
        self.vector2 = Vector([0,-5], [0,5])
        wVector1 = VectorSlider(self.vector1, self.plot_scene, value=[5,5],min=-10.0, max=10.0, step=0.1,
                                description="[RED] Vector1", continuous_update=True, orientation="horizontal",
                                readout_format=".1f")
        wVector2 = VectorSlider(self.vector2, self.plot_scene, value=[-5,5],min=-10.0, max=10.0, step=0.1,
                                description="[BLUE] Vector2", continuous_update=True, orientation="horizontal",
                                readout_format=".1f")
    
        ##################################################
        # Alpha Slider
        ##################################################
        self.alpha = [0.4]
        wAlpha = Slider(self.alpha, 0, self.plot_scene, value=0.4, min=-0.5, max=1.5, step=0.1,
                        description="Alpha: ", readout_format=".1f")
        
        ##################################################
        # SCORE
        ##################################################
        self.question = [0]
        wQ1 = PlayButton(self.question, 0, self.plot_scene, value=0, min=0, max=100, step=1,
                            description="Animation Control")
        scorer = ScoreBox(SCORES, 0, self.plot_scene, value=0, min=0, max=10, step=1, description="Score: ")


        ##################################################
        # QUESTION
        ##################################################
        question1_description = f""" <font size="{self.font_size}">
                                    When select the <b>Question 1</b> in Concept options, try to play with Alpha slidebar, what do you see? <br>
                                    When <b>Question 1 Animation</b> is selected, click the <b>play</b> button right below, alpha will change automatically for visualization.<br>
                                    How to <b>prove</b> alpha * V1[RED] + (1-alpha) * V2[BLUE] == V3[ORANGE]?
                                </font>"""
        question1 = QuestionBox("Question 1", question1_description)
        
        
        module1_description = f""" <font size="{self.font_size}">
                                This section covers the Addition, Cross Product, Vector Projection, and has a question at the end. <br>
                                Try to review the concepts, and answer the question, then enter a self reported score in the Score Box (0 - 10 with 10 being perfect). <br>
                                <b>Instruction</b>: select a concept, then try to play with the slidebars on the right, see how the demo changes, have fun!
                              </font>"""
        
        module1 = QuestionBox("Description", module1_description)
        
        items = [widgets.VBox([wOptions.button, wQ1.button]), 
                 widgets.VBox([wVector1.slider, wVector2.slider, wAlpha.slider, scorer.score])]
        whbox = widgets.VBox([module1.text, widgets.HBox(items), question1.text])
        self.whbox = whbox
    
    def init_figure(self, fig):
        ##################################################
        # Init Figure
        ##################################################
        self.fig = fig
        self.fig.clf()
        self.plot_scene()
    
        
    def plot_scene(self):
        
        self.fig.clf()
        if self.concept[0] == self.options[0]:
            plot_two_vectors(self.fig, self.vector1, self.vector2)
        elif self.concept[0] == self.options[1]:
            plot_vector_addition(self.fig, self.vector1, self.vector2)
        elif self.concept[0] == self.options[2]:
            plot_vector_cross_product(self.fig, self.vector1, self.vector2)
        elif self.concept[0] == self.options[3]:
            plot_vector_projection(self.fig, self.vector1, self.vector2)
        elif self.concept[0] == self.options[4]:
            plot_vector_question1(self.fig, self.alpha[0], self.vector1, self.vector2)
        elif self.concept[0] == self.options[5]:
            plot_vector_question1_animation(self.fig, self.question[0], self.vector1, self.vector2)
    
    
    def __call__(self, fig):
        self.init_figure(fig)
        display(self.whbox)
    
        
