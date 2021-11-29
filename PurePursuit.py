import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


#Description: The pure pursuit algorithm

class Pursuit:
    """
    An implementation of the pure pursuit algorithm
    """

    def __init__(self, state):
        """
        Default constructor

        param state: The state vector of the vehicle at time t
        """

        #----------Image Variables----------#
        #The perspective warped image with virtual space
        self.virtual_img = state.virtual_img

        #The path coordinates in pixels
        self.pic_pixy = state.pic_pixy  #The path y coordinates in the picture frame in pixels
        self.pic_pixx = state.pic_pixx #The path x coordinates in the picture frame in pixels

        #The origin of the vehicle frame
        self.vorigin_y = state.vorigin_y
        self.vorigin_x = state.vorigin_x

        #The pixel to meter transformation
        self.x_p2m = state.x_p2m
        self.y_p2m =state.y_p2m
        
        #----------State Variables----------#
        #The time since program initatiation
        self.t = state.t

        #The vehicle velocity
        self.vel = state.velocity

        #The path (Vectors of x and y coordinates)
        self.pathy = state.pathy
        self.pathx = state.pathx

        #The vehicle length (front axel to rear axel)
        self.vlength = state.vlength

        #----------Pure Pursuit Parameters----------#
        #The proportional lookahead gain
        self.k_lkahead = 0.75

        #The lookahead distance
        self.d_lkahead = None

        #The index of the lookahead point
        self.target_ind = None

        #The radius of the steering circle
        self.str_rad = None

        #The direction of the steering circle
        self.direction = None

        #The steering angle
        self.steering_angle = None


    def calc_lkahead(self, velocity=None):
        """
        Calculates the lookahead distance

        param velocity: The velocity of the vehicle
        """

        if velocity is None:
            velocity = self.velocity
        
        self.d_lkahead = self.k_lkahead * velocity

    def lkahead(self, d_lkahead=None, plot=False):
        """
        Searches the path for the point one lookahead distance away from the vehicle (if multiple points exist, choose the one further along the path)

        param d_lkahead: The lookahead distance
        param plot: a boolean of whether to plot the result
        return: The index of the target point
        """

        if d_lkahead is None:
            d_lkahead = self.d_lkahead

        #Calculate the distance to the vehicle refrence point for each point on the path
        ysq = np.power(self.pathy, 2)
        xsq = np.power(self.pathx, 2)
        distances = np.sqrt(xsq + ysq)

        #Calculate how close each point is to the lookahead distance
        differences = np.abs(distances - d_lkahead)

        cond = False

        while(cond==False):

            #Get the index of the path point closest to the lookahead distance
            target_ind = np.argmin(differences)  #TODO: Determine how argmin decides between equally small values

            #Check to see if the target is ahead of the vehicle, if not eliminate it as a posibility
            if(self.pathy[target_ind] > self.vlength):
                cond = True
            else:
                differences[target_ind] = 1000

        self.target_ind = target_ind

        if plot == True:
            figure, axis1 = plt.subplots(1,1)
            axis1.imshow(self.virtual_img)
            axis1.plot([self.vorigin_x, self.pic_pixx[self.target_ind]], [self.vorigin_y, self.pic_pixy[self.target_ind]], marker = 'o')
            axis1.plot(self.pic_pixx, self.pic_pixy)
            #(self.pic_pixx[self.target_ind], self.pic_pixy[self.target_ind])
            axis1.set_title("Lookahead Line")
            plt.show()
        
        return target_ind

    def calc_circ_rad(self, t_ind, plot=False):
        """
        Calculates the radius of the pure pursuit steering circle

        param t_ind: The index of the target point
        return: The radius of the pure pursuit steering circle
        """

        #Calculate the angle between the heading of the vehicle and the lookahead line
        alpha = np.arctan(self.pathx[t_ind] / self.pathy[t_ind])

        #Calculate the radius of the steering circle
        str_rad = self.d_lkahead / (2*np.sin(alpha))
        self.str_rad = str_rad

        #Determine the direction (right or left) of the steering circle
        if (self.pathx[self.target_ind]>0):
            direction = 1
        else:
            direction = 0

        self.direction = direction

        if plot == True:
            if direction:
                corigin = self.vorigin_x + (str_rad // self.x_p2m)
            else:
                corigin = self.vorigin_x + (str_rad // self.x_p2m)

            figure, axis1 = plt.subplots(1,1)
            axis1.imshow(self.virtual_img)
            str_circle = Ellipse((corigin, self.vorigin_y), width=2 * (str_rad // self.x_p2m), height=2 * (str_rad // self.y_p2m), fill=False, color='red')
            axis1.plot([self.vorigin_x, self.pic_pixx[self.target_ind]], [self.vorigin_y, self.pic_pixy[self.target_ind]], marker = 'o')
            axis1.plot(self.pic_pixx, self.pic_pixy)
            axis1.add_artist(str_circle)
            axis1.set_title("Lookahead Line and Steering Circle")
            plt.show()

        return str_rad, direction

    def calc_steering_angle(self, str_rad):
        """
        Calculates the pure pursuit steering angle

        param str_rad: The radius of the pure pursuit steering angle
        return: The pure pursuit steering angle
        """

        steering_angle = np.arctan((1 / str_rad)*self.vlength)

        self.steering_angle = steering_angle

def pipeline(state):

    ppframe = Pursuit(state)
    
    lookahead_distance = ppframe.calc_lkahead(13)

    lookahead_ind = ppframe.lkahead(lookahead_distance, plot=True)

    circle = ppframe.calc_circ_rad(lookahead_ind, plot=True)

    steering_angle = ppframe.calc_steering_angle(circle[0])

    return steering_angle, ppframe.direction




    
    









    

        
        
    

        

        
        








