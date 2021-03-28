# CRWutils.py
''' Functions for running correlated random walks

    Dave Hunt
    University of Washington
    February 28, 2021
'''

# rev 1.0	Dev

import numpy as np
from numpy import random  as rand

import math
import random
from random import randrange

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker

import scipy as scipy
from scipy import optimize
from scipy import stats
from scipy.stats import t
from scipy.stats import wrapcauchy

################# VARIABLES ########################

# # The walks occur within a 2-dimensional array, or grid, x wide by y high
gridsize_X = 0                                # sets the grid size x and y
gridsize_Y = 0

# GLOBALS
startX = 0                                    # Home x - set the starting point of each walk
startY = 0                                    # Home y

steps = np.zeros((1,2))                       # A numpy array (1 row, 2 columns) to record steps visited, initially set to zeros, used to draw walk lines
stepBreak = []                                # An index into steps for recording the end of each walk, used to break out individual walk lines
stepCount = 0                                 # Keeps track of how many steps have been taken in a particular walk

srcNum = 1                                    # At this point, a carry-over from versions that tested walks with multiple sources, each given a unique number. 
                                              # Here srcNum (source ID) will always be 1, even for a patch covering multiple cells

# The eight possible movement directions for CRW
# This model ignores "stay in place" as an option as it does not affect the likelihood of encounter
directions8 = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])

# A grid is created that simulates lithic source points. Each lithic source has a unique source ID (SrcNum). 
# As walkers encounter each lithic source, that source is recorded and the encounter tallied.
# In this version, there will only be a single patch but patches greater than 1x1 will cover multiple cells
lithics = np.zeros((0,0))                     # The lithics grid
encounteredLithics = 0                        # Tracks number of lithic sources encountered each walk, allows <=1 in this version
sourceReference = []                          # A list to track the srcNum, x and y coordinates, and various attributes like distance from Home, and tally
prob_list = ()                                # The biased distribution of 8 possible moves on the grid
new_cell = randrange(1,9)                     # Used to hold the next cell to move to, initially set to 1 of 8 random directoins for the first step
tiny_number = 0.0000000001
almost_One =  0.99

def initialize_grid (x, y):
    """ Initialize the grid and related attributes

    Args:
        x (int): the x dimension of the grid
        y (int): the y dimension of the grid
    """
    global steps                # global declarations
    global lithics
    global startX
    global startY
    global gridsize_X
    global gridsize_Y

    gridsize_X = x              
    gridsize_Y = y
    steps = np.zeros((1,2))                       # a numpy array to hold all the steps (xy pairs)
    lithics = np.zeros((gridsize_X,gridsize_Y))   # a numpy array, grid-sized, to represent patch locations
    startX = round(gridsize_X / 2)                # Home x - set the starting point of each walk
    startY = round(gridsize_Y / 2)                # Home y
    steps[0,0] = startX                           # The first step or 'home' is set to center of grid.
    steps[0,1] = startY
    return

def get_prob_list(rho):
    """ Given a value, rho, generate the wrapped Cauchy distribution and then 
        slice it into portions that represent the 8 nearest neighbor (nn) cells
        surrounding the current cell

        # nearest neighbor cells labelled as:
        #      H   A   B
        #      G   -   C
        #      F   E   D

    Args:
        rho ([float]): Value from 0 (no directional bias) to 1
    
    Returns:
        [tuple]: the directionally-biased probability list
    """
    slices = 8                          # is a 2D grid system, slices will always be 8, 8 nearest neighbors
    tiny_number = 0.0000000001          # avoid divide by zero
    almost_one =  0.99                  # avoid infinity
    
    if rho <= 0:                        # if rho is either 0 or 1
        rho = tiny_number               # we need to substitute in a "near" value
    if rho >= 1:                        # to avoid calculation errors
        rho = almost_one

    # determine the range/linspace
    x = np.concatenate([np.linspace(np.pi, 2*np.pi, (slices+1)), np.linspace(0, np.pi, (slices+1))], axis=0)

    wcc = wrapcauchy.cdf(x, rho)        # calc the wrapped Cauchy cumulative density 

    # wcc[0] - wcc[8] represent one half of the distribution -- from the mode to the anti-mode
    # We essentially have 16 segments, 8 for each side of the distribution
    # We want 8 segments with segment nnA centered on the mode (for forward direction)
    nnA = (wcc[8] - wcc[7]) * 2         # segment A is the difference in cumulative density btwn wcc[8] and [7] * 2
    nnB = wcc[7] - wcc[5]               # B is the cdf from [7] to [5]
    nnC = wcc[5] - wcc[3]               # C is the cdf from [5] to [3]
    nnD = wcc[3] - wcc[1]               # D is the cdf from [3] to [1]
    nnE = (wcc[1] - wcc[0]) * 2         # E is the anti-mode, opposite of A
    nnF = nnD                           # F, G, and H are all the mirror image of D, C, and B, respectively
    nnG = nnC
    nnH = nnB

    problist = (nnA, nnB, nnC, nnD, nnE, nnF, nnG, nnH)  # build the prob_list from the segments
    return(problist)

def roll(massDist):
    """ returns a roll of the die biased by a probability list
    # Solution by scopessuckM8, January 2, 2020
    # https://stackoverflow.com/questions/479236/how-do-i-simulate-biased-die-in-python

    Args:
        massDist (tuple): the shifted probability list (shift in forward direction)

    Returns:
        [int]: next cell to travel to
    """ 
    randRoll = random.random()           # in [0,1)
    sum = 0
    result = 1
    for mass in massDist:
        sum += mass
        if randRoll < sum:
            return result
        result+=1

def place_patch(lithics, patch_size, x, y):      
    """ Places one patch, of given size, in a specific place in the lithics grid
    
    Parameters:
    lithics (numpy array): the numpy array representing lithic-filled cells
    patch_size (int): the patch side size (patch_size x patch_size), for example 3x3
    x (int): x position to place the lower left corner of the patch
    y (int): y position to place the lower left corner of the patch
    
    Returns:
    --
    """
    global srcNum
    lithics[startX, startY] = -1                      # hold a spot in the map center for the walk start by setting to -1, needed in multi-patch version
    dist_holder = []                              

    if ((x != startX) and (y != startY)):             # don't let a patch be set at Home
        clear = True
        for a in range(0, patch_size):                # make sure all the cells for the new patch are empty
            for b in range(0, patch_size):
                if lithics[x+a, y+b] != 0:
                    clear = False
        if clear == True:                             # if the spot is clear, create a patch and record 
            for a in range(0, patch_size):            
                for b in range(0, patch_size): 
                    lithics[x+a, y+b] = srcNum
                    dist_holder.append(get_distance(startX, startY, x+a, y+b)) #calc distance to each cell in the patch

            dist_holder.sort()                        # sort the individual cell distances
            dist = dist_holder[0]                     # use the distance closest to Home
            tally=0       
            step_sum = 0
            total_steps = 0                           # save the src info 
            sourceReference.append([srcNum, x, y, dist, tally, patch_size, step_sum, total_steps]) #srcNum, x coord, y coord, distance from home, encounter tally, patch size, step_sum this walk, total steps this set of walks
    
    lithics[startX, startY] = 0                       # reset the start point back to zero now the patch is placed
    return

def take_step_CRW(prev_cell):
    """Determines the next step in the CRW walk and follows
       the probabilities of selection per the prob_list order (0 .. 7)

    Args:
        prev_cell (int): the nearest neighbor grid cell that the walker
        is coming from
        # nearest neighbor cells numbers as 
        #      8   1   2
        #      7   -   3
        #      6   5   4

    Returns:
        [int]: the nearest neighbor # for the next cell step
    """
    global steps

    forward_cell = find_opposite_cell(prev_cell)      # determine the forward motion cell
    new_cell = roll(rotate(prob_list,(forward_cell))) # get the next move cell         
    move = directions8[new_cell-1]                    # convert to grid x, y directions
    nextloc = [steps[-1] + move]                      # append the move to the steps list
    steps = np.append(steps, nextloc, axis=0)         # save the step
    return new_cell

def find_opposite_cell(previous_cell):
    """ Finds the cell that represents forward motion
    When a CRW enters a cell, this function takes the previous cell
    and determines which nearest neighbor cell of the current cell is 
    opposite from it.  This opposite cell represents forward motion/
    directional persistence.  The Cauchy distribution is then centered on this cell

    The nearest neighbors are numbered like this:

          8   1   2
          7   -   3
          6   5   4
    Args:
        previous_cell (int): the grid cell (number as above) we came from

    Returns:
        forward_cell (int): the cell that would be next if we kept going straight
    """
    if previous_cell < 5:
        forward_cell = previous_cell + 4
    else:
        forward_cell = previous_cell - 4
    return forward_cell

def rotate(prob_list, forward_cell):
    """Rotates the probability list to "face" the direction forward
    after a step

    Args:
        prob_list (list): the Cauchy distribution in octants
        forward_cell (int): the cell that represents forward motion

    Returns:
        list: the rotated prob_list
    """
    rotate_n = 1 - forward_cell   # rotating to the forward position index
    return prob_list[rotate_n:] + prob_list[:rotate_n]

def take_first_encounter_walk (num_walks, num_of_steps):
    """ The main walk driver, loops on number of walks for number of steps.

    Args:
        num_walks (int): number of walks to take
        num_of_steps (int): number of steps per walks

    Returns:
        No return but appends to the global steps list
    """  
    global steps                     # records all the steps of all the walks
    global encounteredLithics        # keeps a record of encounters
    global stepBreak                 # the step index, indicates where each walk starts and ends
    global new_cell                  # the next cell to enter
    
    for walk in range(num_walks):            # begin each independent walk
        out_of_bounds = False                              # flag to ensure we're in bounds of the grid
        encounter = False                                  # flag that we haven't yet encountered a patch
        last_source = 0                                    # in walks looking at multiple sources, this prevents repeat visit counts
        stepCount = 0                                      # reset the stepCount for each individual walk   
        new_cell = randrange(1,9)
        while ((stepCount < num_of_steps) and (out_of_bounds == False) and (encounter == False)):
            stepCount += 1

            new_cell = take_step_CRW(find_opposite_cell(new_cell))     # advance one step

            if out_of_bounds_check(steps[-1,]):                        # make sure we haven't stepped out of the grid...
                steps = steps[:-1]                                     # slice out that last, out-of-bounds step
                stepCount = stepCount - 1                              # and decrement the stepCount
                out_of_bounds = True                                   # if we go out of bounds, the walk stops
            else:  
                SrcNum = check_for_lithics(steps[-1,])                                        # see if there is a lithic source at this location
                if ((SrcNum != 0) and (SrcNum != last_source)):                               # if we have a lithic source and it isn't the same as last time...
                    encounteredLithics += 1                                                   # record the encounter 
                    encounter = True                                                          # we hit a patch
                    sourceReference[SrcNum-1][4] = sourceReference[SrcNum-1][4] + 1           # inc the previous tally for the specific source      
                    sourceReference[SrcNum-1][6] = sourceReference[SrcNum-1][6] + stepCount   # add the current stepcount to the sum                                
                    
                last_source = SrcNum                                   # update the last_source to the current source 
        sourceReference[SrcNum-1][7] = sourceReference[SrcNum-1][7] + stepCount
        stepBreak.append(len(steps))                                   # record the last step for this walk line
            
        reset = np.array([[startX,startY]])                            # reset the starting point to end of steps to start new walk
        steps = np.append(steps, reset, axis=0)  
    return

def check_for_lithics (step):
    """ Check if this point in the matrix contains a lithic source.
        If so, return the resource value, this will be used to record the encounter

    Args:
        step ([tuple]): x, y coordinates of the current cell

    Returns:
        [int]: The SrcNum, ID of the source encountered 
    """
    srcNum = 0
    if lithics[int(step[0]), int(step[1])] > 0:           # any value >0 is a lithic source
        srcNum = lithics[int(step[0]), int(step[1])]
    return int(srcNum)

def out_of_bounds_check(newpt):
    """ checks if the next step will move the walker off the grid
        If true, the walk will be terminated (no rebound)

    Args:
        newpt ([type]): the point we might move to if OK

    Returns:
        [Boolean]: confirms if OK to move there
    """
    if ((newpt[0] < 0) or (newpt[0] > (gridsize_X-1))) or ((newpt[1] < 0) or (newpt[1] > (gridsize_Y-1))):
        return True
    else:
        return False

def get_distance(x1, y1, x2, y2):
    """ Determines the distance between two grid cells

    Args:
        x1 (int): x coordinate for cell 1
        y1 (int): y 1
        x2 (int): x 2
        y2 (int): y 2

    Returns:
        float: distance in grid cells, rounded to 2 decimals
    """
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 
    return round(dist,2)

def plot_lithics(lithics):
    """ Plots the location of the lithic patch onto the final walk plot

    Args:
        lithics ([numpy array]): 2D array representing the walk grid, lithic patches are non-zero cells
    """
    i = 0
    j = 0
    for i in range(gridsize_X):                                  # find any patches in the grid
        for j in range(gridsize_Y):
            if lithics[i,j] > 0:
                plt.scatter(i,j, c='k', marker="s", s=30)        # places a black square to represent the lithic
    return 

def plot_walk(label, num_of_walks, show_lithics=True):
    """ Plots the random walk

    Args:
        label (string): descriptive string for the walk attributes, incorporated into the title and filename
        num_of_walks (int): number of walks represented in this plot
        show_lithics (Boolean): shows patch in plot (T/F)
    """
    # walk plot attributes
    nrows = 2
    ncols = 3
    fig = plt.figure(figsize=(16,9))
    grid = GridSpec(nrows, ncols, left=0.1, bottom=0.15, right=0.94, top=0.94, wspace=.5, hspace=0.3)
    ax1 = fig.add_subplot(grid[0:2,0:2])
   
    if show_lithics == True:
        plot_lithics(lithics)                                  # plot the lithics onto the graph
     
    stepStart = 0                                           # draw all the walk lines  
    for z in stepBreak:                                     # use the stepBreak index to divide out the individual walk lines for plotting
        xdata = steps[stepStart:z-1, 0]
        ydata = steps[stepStart:z-1 ,1]
        # c=next(cycol)                                     # optional color version of walk plot
        # line1, = ax1.plot([], [], lw=1.2, c=c)   
        line1, = ax1.plot([], [], lw=1.2, color='darkgray') # creates gray version of walk plot  
        line1.set_data(xdata, ydata)                        # create the individual walk line
        stepStart = z

    ax1.plot(steps[0,0], steps[0,0], 'rs', ms=7, markeredgecolor='black', markeredgewidth=2)        # home marker   
    
    ax1.set_title('Correlated Random Walk: ' + label + ' -- Total steps: %d' % (len(steps) - num_of_walks - 1), fontsize=16)
       
    ax1.set_facecolor('#fffffa')  
    ax1.set_ylim(-0.5, ((gridsize_X) + 0.5))
    ax1.set_xlim(-0.5, ((gridsize_Y) + 0.5))

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(1))
    ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
    ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)
    
    plt.show()
    
    rando = random.randrange(100)                                                     # optional for unique filename
    fig.savefig('randWalk_' + label + '_' +str(rando) + '.png', bbox_inches='tight')  # save each plot image
    return

def reset_variables():
    """ Resets all the variables between walk sets
    """

    global stepCount
    global steps
    global stepBreak
    global lithics
    global encounteredLithics
    global sourceReference
    global srcNum
    global new_cell
    
    stepCount = 0                                 # keeps track of how many steps have been taken in a particular walk
    steps = np.zeros((1,2))                       # A numpy array (1 row, 2 columns) to record steps visited 
    stepBreak = []
    steps[0,0] = startX                           # The first step or 'home' is set to center of grid.
    steps[0,1] = startY                           #    on all the random walks, initially set to zeros, used to draw walk lines
    srcNum = 1
    new_cell = randrange(1,9)

    # A parallel grid is created that simulates lithic source points. Each lithic source has a unique source ID (SrcNum). 
    # As walkers encounter each lithic source, that source is recorded and the encounter tallied.
    lithics = np.zeros((gridsize_X,gridsize_Y))   # the lithics grid
    encounteredLithics = 0                        # tracks number of lithic sources encountered each walk
    sourceReference = []                          # a list to track the srcNum, x and y coordinates, distance from Home, and tally
    return

def exponential(x, a, k, b):
    """Used to calculate exponential decay curve formula, conforms with y = Ae^Bx
    """
    return a*np.exp(x*k) + b            # TODO need dtype=np.float128

def linear(x, m, b):
    """Returns a line with equation mx+b
    """
    return m*x + b

def regression_components (name, x, y):
    """Calculates the various components needed for slope and intercept comparisons
       Based on Zar, J., (2010), Biostatistical Analysis, 5th Ed., Pearson
       Chap 18, "Comparing Two Slopes", pg. 387 - 391, equations 1 - 4

    Args:
        name (string): a lable to name the dataset
        x (nparray): the independent variable list/arry
        y (list): the dependent list

    Returns:
        dict: dictionary of all the components needed for compare_slopes and compare_intercepts
    """   
    n = len(x)
    X_bar = np.mean(x)
    sum_X = sum(x)
    sum_X2 = sum([i*i for i in x])
    sum_x2 = sum_X2 - (sum_X)**2 / n
    Y_bar = np.mean(y)
    sum_Y = sum(y)
    sum_Y2 = sum([i*i for i in y])
    sum_y2 = sum_Y2 - sum_Y**2 / n          # same as total SS

    XY = []
    for i, j in zip(x, y):
        XY.append(i * j)
    sum_XY = sum(XY)

    sum_xy = sum_XY - (sum_X * sum_Y)/n
    reg_SS = sum_xy**2/sum_x2
    b = sum_xy / sum_x2                    #slope
    a = Y_bar - b * X_bar
    resid_SS = sum_y2 - sum_xy**2 / sum_x2
    resid_DF = n - 2
      
    dict = {
        "name": name,
        "n": n,
        "X_bar": X_bar,
        "sum_X": sum_X,
        "sum_X2": sum_X2,
        "sum_x2": sum_x2,
        "Y_bar": Y_bar,
        "sum_Y": sum_Y,
        "sum_Y2": sum_Y2,
        "sum_y2": sum_y2,
        "sum_XY": sum_XY,
        "sum_xy": sum_xy,
        "reg_SS": reg_SS,
        "b": b,
        "a": a,
        "resid_SS": resid_SS,
        "resid_DF": resid_DF 
    }
    return dict    

def compare_slopes (d1, d2, alpha):
    """ Tests slopes of 2 lines with the Ho: β1 = β2 (sampling same population slope) 
        
        Based on Zar, J., (2010), Biostatistical Analysis, 5th Ed., Pearson
        Chap 18, "Comparing Two Slopes", pg. 387 - 391, equations 1 - 4

    Args:
        d1 (dict): Dictionary of regression components for line 1
        d2 (dict): Dictionary of regression components for line 2
        alpha (float): level of significance
    """
    s2YX = (d1['resid_SS'] + d2['resid_SS']) / (d1['resid_DF'] + d2['resid_DF'])  # see Zar
    sb1minb2 = math.sqrt((s2YX/d1['sum_x2']) + (s2YX / d2['sum_x2']))   
    t = abs((d1['b'] - d2['b']) / sb1minb2)  
    p_val = stats.t.sf(t, 29) * 2     # p value
    pp = 1 - (alpha/2)           # two-tailed test, so divide alpha

    v = d1['n'] + d2['n'] - 4    # degrees of freedom
    t0 = stats.t.ppf(pp, v)      # critical value for Student t test, two-tailed    
       
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")   # build the hypothesis string, substring chars, results
    v_sub = str(v).translate(SUB)              
    alpha_sub = str(alpha).translate(SUB)         
    title = d1['name'] + " vs " + d2['name']
    print("Slope comparison for significance: " + title ) 
    print('\tTesting H\N{SUBSCRIPT ZERO}: \u03B2\N{SUBSCRIPT ONE} = \u03B2\N{SUBSCRIPT TWO}')
    print("\tt = " + '{:f}'.format(t) + ' with v (df) = ' + str(v) +  ' and p-value = ' + '{:f}'.format(p_val) )     
    print('\tt' + str(alpha_sub) + '\u208D\N{SUBSCRIPT TWO}\u208E,' + str(v_sub) + ' = ' + '{:f}'.format(t0))    
    
    if t > t0:                    # if t exceeds the critical value, reject Ho
        print("\tH\N{SUBSCRIPT ZERO} is rejected")
    else:
        print("\tH\N{SUBSCRIPT ZERO} is not rejected")     # if Ho is not rejected, try the intercept test
        compare_intercepts(d1, d2, alpha)                  # test intercepts
    return

def compare_intercepts (d1, d2, alpha):
    """ Test slope elevation with the Ho that the lines are parallel.

        Based on Zar, J., (2010), Biostatistical Analysis, 5th Ed., Pearson
        Chap 18, "Comparing Two Elevations", pg. 391-395, equations 12 - 18    

    Args:
        d1 (dict): Dictionary of regression components for line 1
        d2 (dict): Dictionary of regression components for line 2
        alpha (float): level of significance
    """
    
    Ac = d1['sum_x2'] + d2['sum_x2']   # see Zar for formula construction
    Bc = d1['sum_xy'] + d2['sum_xy'] 
    Cc = d1['sum_y2'] + d2['sum_y2'] 
    bc = Bc / Ac
    SSc = Cc - (Bc**2/Ac)
    DFc = d1['n'] + d2['n'] - 3
    s2YXc = SSc / DFc
      
    t_top = (d1['Y_bar'] - d2['Y_bar']) - bc*(d1['X_bar'] - d2['X_bar'])
    t_bott1 = (1/d1['n'] + 1/d2['n']) + (d1['X_bar'] - d2['X_bar'])**2/Ac
    t_bottom = math.sqrt(s2YXc * t_bott1)
    t = abs(t_top / t_bottom)
    
    p_val = stats.t.sf(t, 29) * 2      # p value    
    pp = 1 - (alpha/2)                 # two-tailed test, so divide alpha

    v = d1['n'] + d2['n'] - 4          # degrees of freedom
    t0 = stats.t.ppf(pp, v)            # critical value for Student t test, two-tailed    
       
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")   # build the hypothesis string, substring chars, results
    v_sub = str(v).translate(SUB)         
    alpha_sub = str(alpha).translate(SUB)       
    title = d1['name'] + " vs " + d2['name']          
    print("Intercept/elevation comparison for significance: " + title ) 
    print('\tTesting H\N{SUBSCRIPT ZERO}: The two population regression lines have the same elevation')    
    print("\tt = " + '{:f}'.format(t) + ' with v (df) = ' + str(v) +  ' and p-value = ' + '{:f}'.format(p_val) )     
    print('\tt' + str(alpha_sub) + '\u208D\N{SUBSCRIPT TWO}\u208E,' + str(v_sub) + ' = ' + '{:f}'.format(t0))    

    if t > t0:                         # if t exceeds the critical value, reject Ho
        print("\tH\N{SUBSCRIPT ZERO} is rejected")
    else:
        print("\tH\N{SUBSCRIPT ZERO} is not rejected")
    return

def printDecayGraph (dist, tally, label):
    """ Fits the data to an exponential decay line and plots it

    Args:
        dist (list): x variables, independent, distance from Home to patch
        tally (list): y variables, dependent, tally of encounters with the patch during 100 walks
        label (string): patch name to title the graph and create the filename
    """
   
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(dist, tally, "bo", markersize=5)       # plot the points

    #   popt_exponential: this contains the fitting parameters
    #   pcov_exponential: estimated covariance of the fitting paramters, not used here
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, dist, tally, p0=[1,-0.5, 1])
    a = popt_exponential[0]
    b = -popt_exponential[1]
    c = popt_exponential[2]
    x = np.linspace(0, 140, 100, endpoint = True)
    y = (a * np.exp(-b*x)) + c

    plt.plot(x, y, '-r', linewidth=2, label=r'$y = %0.2fe^{-%0.2fx} + %0.2f$' % (a, b, c))    # plot the line

    # various matplotlib settings to visualize the graph
    ax1.set_xlim(-0.5,(max(dist))+5)
    ax1.set_ylim(-0.5,(max(tally))+5)
    
    ax1.set_title("CRW " + label + ": Encounters vs. Distance", fontsize=22)
    ax1.set_xlabel("Distance (grid cells)",  fontsize=18)
    ax1.set_ylabel("Encounters (n)",  fontsize=20)
    ax1.legend(loc='best', fontsize=16)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.tick_params(axis='both',which='major', direction="out", bottom="on", length=8, labelsize=14)
    ax1.tick_params(axis='both',which='minor', direction="out", bottom="on", length=5, labelsize=12)

    fig.tight_layout() 
    plt.savefig("output/CRW_" + label + '_exp_decay.png')         # saves graph to /output subfolder

def wrapped_Cauchy(rho):
    """Creates a wrapped Cauchy distribution for plotting

    Args:
        rho (float): The concentration parameter for the distribution
    """
    x = np.concatenate([np.linspace(np.pi, 2*np.pi, 100), np.linspace(0, np.pi, 100)], axis=0)
    
    wrapped_cauchy_pdf = wrapcauchy.pdf(x, rho)
    x[:100] = x[:100] - 2*np.pi

    if rho == 0:
        plt.plot(x, [1/(2*np.pi)]*len(x), label='$\\rho$ = ' + str(rho), lw=2.5)
    else:
        plt.plot(x, wrapped_cauchy_pdf, label='$\\rho$ = %.2f' % rho, lw=2.5)


def printSemiLogGraph(dist, tally, label):
    """ Fits the data to a line and plots it

    Args:
        dist (nparray): log x variables, independent, distance from Home to patch
        tally (list): y variables, dependent, tally of encounters with the patch during 100 walks
        label (string): patch name to title the graph and create the filename
    """
  
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])
    
    ax1.plot(np.log(dist), tally, "bo", markersize=5)    

    #   popt_linear: this contains the fitting parameters
    #   pcov_linear: estimated covariance of the fitting paramters, not used here
    popt_linear, pcov_linear = scipy.optimize.curve_fit(linear, np.log(dist), tally, p0=[((75-25)/(44-2)), 0])
    perr_linear = np.sqrt(np.diag(pcov_linear))
    print ("slope = %0.2f (+/-) %0.2f" % (popt_linear[0], perr_linear[0]))
    print( "y-intercept = %0.2f (+/-) %0.2f" %(popt_linear[1], perr_linear[1]))

    # various matplotlib settings to visualize the graph
    ax1.plot(np.log(dist), linear(np.log(dist), *popt_linear), 'k--', label="y= %0.2fx + %0.2f" % (popt_linear[0], popt_linear[1]))

    # ax1.set_xlim(-0.5,(max(p1['distance']))+5)
    # ax1.set_ylim(-0.5,(max(p1['tally']))+5)

    ax1.set_title("CRW " + label + ": Encounters vs. Distance (ln)", fontsize=22)
    ax1.set_xlabel("Distance (ln)",  fontsize=18)
    ax1.set_ylabel("Encounters (n)",  fontsize=20)
    ax1.legend(loc='best', fontsize=16)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.tick_params(axis='both',which='major', direction="out", bottom="on", length=8, labelsize=14)
    ax1.tick_params(axis='both',which='minor', direction="out", bottom="on", length=5, labelsize=14)

    fig.tight_layout()
    plt.savefig("output/CRW_" + label + '_semi_log.png')    # saves graph to /output subfolder





    