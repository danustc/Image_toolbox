'''
Signal plot for \Delta F/F. Created by Dan on 07/27/2017.
Last modification:
'''
import matplotlib.pyplot as plt
import numpy as np

def nature_style_dffplot(dff_data, dt = 0.5, sc_bar = 0.25):
    """
    Present delta F/F data in nature style
    """
    n_time, n_cell = dff_data.shape
    tt = np.arange(n_time)*dt

    tmark = -dt*10

    fig = plt.figure(figsize = (7.0*n_time/500,10))
    for ii in xrange(n_cell):
        dff = dff_data[:,ii]
        ax = fig.add_subplot(n_cell,1, ii+1)
        ax.plot(tt, dff)
        ax.plot([tmark,tmark], [0, sc_bar], color = 'k', linewidth = 3)
        ax.set_xlim([-dt*20, tt[-1]])

        ax.set_ylim([-0.05, sc_bar*5])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('time (s)', fontsize = 12)
    tax = fig.get_axes()[0]
    tax.text(tmark, sc_bar*1.5, r'$\Delta F/F = $'+str(sc_bar), fontsize = 12)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0)

    return fig


# Raster plot, color coded
def dff_rasterplot(dff_ordered, dt = 0.5, fw = 7.0, tunit = 'm'):
    '''
    dff_ordered: df_f ordered from most active to least active
    # rows:  # of time points
    # columns: # of cells
    Must be transposed before plotting.
    fw: figure width
    '''
    NT, NC = dff_ordered.shape
    # whether to display in the unit of 10 seconds or 1 min
    if(tunit == 's'):
        time_tick = dt*np.arange(0, NT, 30)
        t_max = dt*NT
        t_label = 'Time (s)'
    elif(tunit == 'm'):
        time_tick = dt*np.arange(0., NT, 60./dt)/60.
        t_max = dt*NT/60.
        t_label = 'Time (min)'

    cell_tick = np.array([1,NC])
    cell_tick_label = ['Cell 1', 'Cell '+str(NC)]

    fig = plt.figure(figsize = (fw, fw*4*NC/NT))
    ax = fig.add_subplot(111)
    rshow = ax.imshow(dff_ordered.T, cmap = 'Greens', interpolation = 'None', extent = [0., t_max, cell_tick[-1], cell_tick[0]], aspect = 'auto')
    ax.set_xticks(time_tick)
    ax.set_yticks(cell_tick)
    ax.set_yticklabels(cell_tick_label)
    ax.tick_params(labelsize = 12)
    ax.set_xlabel(t_label, fontsize = 12)
    cbar = fig.colorbar(rshow, ax = ax, orientation = 'vertical', pad = 0.02, aspect = NC/3)
    cbar.ax.tick_params(labelsize = 12)

    plt.tight_layout()
    return fig