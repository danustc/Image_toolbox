'''
Signal plot for \Delta F/F. Created by Dan on 07/27/2017.
Last modification:
'''
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def compact_dffplot(dff_data, dt = 0.5, ts_bar = 200, sc_bar = 0.25):
    '''
    compact df/f plot.
    '''


def nature_style_dffplot(dff_data, dt = 0.5, sc_bar = 0.25):
    """
    Present delta F/F data in nature style
    """
    n_time, n_cell = dff_data.shape
    tt = np.arange(n_time)*dt

    tmark = -dt*10

    fig = plt.figure(figsize = (7.0*n_time/1500,10))
    for ii in range(n_cell):
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
def dff_rasterplot(dff_ordered, dt = 0.5, fw = 7.0, tunit = 'm', n_truncate = None, title = None, fig = None):
    '''
    dff_ordered: df_f ordered from most active to least active
    # rows:  # of time points
    # columns: # of cells
    Must be transposed before plotting.
    fw: figure width
    '''
    NT, NC = dff_ordered.shape
    if n_truncate is None:
        n_display = NC
    else:
        n_display = np.min([NC, n_truncate])
    print("# of cells displayed:", n_display)
    # whether to display in the unit of 10 seconds or 1 min
    if(tunit == 's'):
        time_tick = dt*np.arange(0, NT, 30)
        t_max = dt*NT
        t_label = 'Time (s)'
    elif(tunit == 'm'):
        time_tick = dt*np.arange(0., NT, 60./dt)/60.
        t_max = dt*NT/60.
        t_label = 'Time (min)'
    elif(tunit == 'h'):
        time_tick = dt*np.arange(0, NT, int(NT/2))
        t_max = dt*NT
        t_label = 'Freq(Hz)'

    cell_tick = np.array([1,n_display])
    cell_tick_label = ['Cell 1', 'Cell '+str(n_display)]
    canvas_scale = int(np.ceil(200./n_display))
    print("fig size:", fw, fw*5*canvas_scale*n_display/NT)
    if fig is None:
        fig = plt.figure(figsize = (fw, fw*5*canvas_scale*n_display/NT))

    ax = fig.add_subplot(111)
    rshow = ax.imshow(dff_ordered[:, :n_display].T, cmap = 'plasma', interpolation = 'None', extent = [0., t_max, cell_tick[-1], cell_tick[0]], aspect = 'auto')
    ax.set_xticks(time_tick)
    ax.set_yticks(cell_tick)
    ax.set_yticklabels(cell_tick_label)
    ax.tick_params(labelsize = 14)
    ax.set_xlabel(t_label, fontsize = 14)
    cbar = fig.colorbar(rshow, ax = ax, orientation = 'vertical', pad = 0.02, aspect = n_display/5)
    cbar.ax.tick_params(labelsize = 14)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig


def spectrogram(sg_gram, trange, krange, k_int = None):
    '''
    plot a spectrogram
    if t_int, plot a time integration of the spectrogram
    if k_int is None, do not plot the integration of spectrogram within the range of k_int.
    '''
    fig_width = 7.
    NK, NW = sg_gram.shape
    dk = krange/NK # the k_steps
    fig = plt.figure(figsize = (fig_width, fig_width*0.62))
    sg_temp = (sg_gram**2).sum(axis = 1) # sum across time

    if k_int is None:
        gs = gridspec.GridSpec(1,2, width_ratios = [1,5])
        gs.update(wspace = 0)
        ax_temp = fig.add_subplot(gs[0])
        ax_sg = fig.add_subplot(gs[1]) # The imshow of spectrogram
        ax_sg.imshow(sg_gram[1:], origin = 'lower', aspect = 'auto', extent = [0, trange, dk, krange]) # show the non-zero frequency components
        sg_int = None
    else:
        ki_, kf_ = int(k_int[0]/dk), int(np.ceil(k_int[1]/dk))
        sg_int = (sg_gram[ki_:kf_]**2).sum(axis = 0)
        gs = gridspec.GridSpec(2,2, width_ratios = [1,6], height_ratios = [1,4])
        gs.update(wspace = 0, hspace = 0)
        ax_int = fig.add_subplot(gs[0,1]) # The k_int plot
        ax_int.set_yticklabels([])
        ax_temp = fig.add_subplot(gs[1,0]) # The k-space sum up
        ax_sg = fig.add_subplot(gs[1,1]) # The imshow of spectrogram
        ax_int.plot(sg_int, '-k')
        ax_int.set_xticklabels([])
        ax_sg.imshow(sg_gram[1:], origin = 'lower', aspect = 'auto', extent = [0, trange, dk, krange]) # show the non-zero frequency components
        ax_sg.plot(np.linspace(0, trange, NW), k_int[0]*np.ones(NW), '--r')
        ax_sg.plot(np.linspace(0, trange, NW), k_int[1]*np.ones(NW), '--r')

    ax_sg.set_yticklabels([])
    ax_sg.set_xlabel('Time (s)', fontsize = 12)
    ax_temp.plot(sg_temp[1:], np.arange(1,NK)*dk, 'k')
    ax_temp.set_ylim([dk, krange])
    ax_temp.invert_xaxis()
    ax_temp.set_xticklabels([])
    ax_temp.set_ylabel('Hz', fontsize = 12)

    plt.tight_layout()
    return fig, sg_temp, sg_int
