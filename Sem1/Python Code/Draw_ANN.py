
import numpy as np
import matplotlib.pyplot as plt

def drawANN(layers):
    
    ## Limits to the display....
    xlim0 = 0
    xlim1 = 8
    ylim0 = 0
    ylim1 = 4

    ## "Filler" space for the display....
    xlimEdge = 0.25
    ylimEdge = 0.25

    ## Midpoint of the y-position of display.
    ylimMid = (ylim0 + ylim1) / 2.0


    ## x-positions....
    xpos = np.linspace(xlim0, xlim1, len(layers))
    ypos = {}

    ## y-positions.
    spacingFactor = 1.8

    ## Calculate drawing parameters.
    nLayers = len(layers)
    maxNeurons = np.max(layers)
    minNeurons = 1

    minSpacing = (ylim1 - ylim0) / (maxNeurons - 1)
    maxSpacing = minSpacing * spacingFactor

    #
    # Colours for the layers.
    # Mimic the colours in Nolan's notebook.
    # This could probably be done differently but it's good enough.
    #

    # Hidden layers
    colours = ['cornflowerblue'] * nLayers

    # Input layer
    colours[0] = 'gold'

    # Output layer
    colours[-1] = 'coral'
    
    #
    # Generate the network drawing for each layer.
    #
    for i in range(0, nLayers):

        ## Number of neurons....
        nNeurons = layers[i]
    
        y = np.arange(0, nNeurons)
        spacing = (nNeurons - minNeurons) / (maxNeurons - minNeurons) * (minSpacing - maxSpacing) + maxSpacing

        y = y * spacing

        ymid = y[-1] / 2.0
        y = y + (ylimMid - ymid)
        ypos[i] = y


    ## Radius of the circles....
    r = 1/3 * minSpacing

    ## Properties for displaying the connection lines....
    lineAlpha = 0.7
    lineColour = (0.7, 0.7, 0.7)
    lineWidth = 0.5

    ## Rotation in degrees....
    xtickRotation = 0 ##-60         

    fig, ax = plt.subplots()

    ## Draw the neurons in all layers.
    for ilayer in range(0, nLayers):
        nNeurons = layers[ilayer]
        for ineuron in range(0, nNeurons):
            circ = plt.Circle((xpos[ilayer], ypos[ilayer][ineuron]),
                              r, fill=True, facecolor=colours[ilayer],
                              lw=lineWidth, edgecolor='k')
            ax.add_artist(circ)


    ## Draw the connections.  Assume full connectivity.
    for ilayer in range(0, nLayers-1):
        for i0 in range(0, layers[ilayer]):
            xy0 = (xpos[ilayer], ypos[ilayer][i0])
            for i1 in range(0, layers[ilayer + 1]):
                xy1 = (xpos[ilayer + 1], ypos[ilayer + 1][i1])
                ax.plot([xy0[0], xy1[0]], 
                         [xy0[1], xy1[1]],
                         color = lineColour,
                         alpha = lineAlpha,
                         linewidth = lineWidth)

            
    ## Set the aspect ratio.	
    ax.set_aspect(1)

    ## Format/hide the axes.
    xticklabels = ['I']
    for i in range(1, nLayers-1):
        xticklabels.append('HL ' + str(i))

    xticklabels.append('O')
    ax.set_xticks(xpos)
    ax.set_xticklabels(xticklabels, rotation = xtickRotation)
    ax.get_yaxis().set_visible(False)

    ax.set_xlim((xlim0 - xlimEdge, xlim1 + xlimEdge))
    ax.set_ylim((ylim0 - ylimEdge, ylim1 + ylimEdge))
    
    return(fig)

