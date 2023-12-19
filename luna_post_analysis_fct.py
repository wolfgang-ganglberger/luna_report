import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
import xml.etree.ElementTree as ET

# read edf eeg data:
from mne.io import read_raw_edf
from mne.filter import filter_data


stage_name_dict = {1: 'N3', 2: 'N2', 3: 'N1', 4: 'R', 5: 'W'}

def read_from_xml(input_path, fs, epoch_length=30, combine_n2n3=False):
    """
    """
    sleep_stage_mapping = {0: 5, 1: 3, 2: 2, 3: 1, 5: 4}

    tree = ET.parse(input_path)
    root = tree.getroot()

    sleep_stages = []
    for sleep_stage in root.iter('SleepStage'):
        sleep_stages.append(sleep_stage_mapping[int(sleep_stage.text)])

    sleep_stages = np.array(sleep_stages)
    sleep_stages = np.repeat(sleep_stages, int(round(fs*epoch_length)))

    return sleep_stages



def plot_spectrogram_so_sp(specs_nopad, data, df_spindle, df_mspindle, df_so,
                            freq, dir_figures, study_id_savename=''):
    
    fontsize_labels = 6
    fontsize_ticks = 6

    start_sec = data['time'].quantile(0)
    end_sec = data['time'].quantile(1)
    data_plot = data.loc[(data['time'] >= start_sec) & (data['time'] <= end_sec)]
    specs_plot = specs_nopad[int(start_sec):int(end_sec), :]

    df_spindle_plot = df_spindle.loc[(df_spindle['start'] >= start_sec) & (df_spindle['stop'] <= end_sec)]
    df_mspindle_plot = df_mspindle.loc[(df_mspindle['msp_start'] >= start_sec) & (df_mspindle['msp_stop'] <= end_sec)]
    df_so_plot = df_so.loc[(df_so['start'] >= start_sec) & (df_so['stop'] <= end_sec)]

    height_ratios = [10, 1, 0.5, 0.5, 1, 1]

    fig, ax = plt.subplots(len(height_ratios), 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': height_ratios})

    # _______________________________________________________________________________
    # plot spectrogram:
    i_axis = 0
    im = ax[i_axis].imshow(specs_plot.T, cmap='turbo', origin='lower', aspect='auto',
                extent=(int(start_sec), int(end_sec), 0, freq.max()),
                vmin=0, vmax=17)
    ax[i_axis].set_yticks([5, 10, 15])
    ax[i_axis].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax[i_axis].set_ylabel("Freq (Hz)", fontsize=fontsize_labels)


    # _______________________________________________________________________________
    # Plot hypnogram:
    i_axis += 1
    palette_stage = ['darkblue', 'blue', 'lightblue', 'purple', 'gold']
    annotations = data_plot['stage'].values.astype(float)
    vals_hypno_tmp = np.zeros((len(annotations), 5))
    vals_hypno_tmp[:] = np.nan
    for i, val in enumerate(annotations):
        if val == 9:
            continue
        vals_hypno_tmp[i, int(val-1)] = 1
        
    vals_hypno_tmp = pd.DataFrame(vals_hypno_tmp, columns=['N3', 'N2', 'N1', 'R', 'W'], index=data_plot['time'])

    vals_hypno_tmp.plot(kind='area', color=palette_stage, alpha=.8, ax=ax[i_axis], sharex=True,
                    stacked=True, lw=0, legend=True)

    ax[i_axis].get_legend().remove()
    ax[i_axis].set_ylim([0, 1])
    ax[i_axis].set_yticks([])
    ax[i_axis].set_ylabel('HYP', labelpad=1, fontsize=fontsize_labels)


    # _______________________________________________________________________________
    # plot spindles:
    i_axis += 1
    for i, row in df_spindle_plot.iterrows():
        ax[i_axis].fill_between([row['start'], row['stop']], [0, 0], [1, 1], color='r', alpha=0.2)
    ax[i_axis].set_ylim([0, 1])
    ax[i_axis].set_yticks([])
    ax[i_axis].set_ylabel('SP', labelpad=1, fontsize=fontsize_labels)

    # _______________________________________________________________________________
    # plot mspindles ('msp_start' and 'msp_stop' columns):
    i_axis += 1
    for i, row in df_mspindle_plot.iterrows():
        ax[i_axis].fill_between([row['msp_start'], row['msp_stop']], [0, 0], [1, 1], color='b', alpha=0.2)
    ax[i_axis].set_ylim([0, 1])
    ax[i_axis].set_yticks([])
    ax[i_axis].set_ylabel('MSP', labelpad=1, fontsize=fontsize_labels)

    # _______________________________________________________________________________
    # plot SOs:
    i_axis += 1
    for i, row in df_so_plot.iterrows():
        ax[i_axis].fill_between([row['start'], row['stop']], [0, 0], [1, 1], color='orange', alpha=0.2)
    ax[i_axis].set_ylim([0, 1])
    ax[i_axis].set_yticks([])
    ax[i_axis].set_ylabel('SO', labelpad=1, fontsize=fontsize_labels)

    # plot q values for start and stop:
    i_axis += 1
    for i, row in df_spindle_plot.iterrows():
        # ax[i_axis].plot([row['start']-0.05, row['stop']+0.05], [float(row['q']), float(row['q'])], c='g', lw=2)
        # make this with fill_between:
        ax[i_axis].fill_between([row['start']-0.01, row['stop']+0.01], float(row['q']-0.01), float(row['q']+0.01), color='g', alpha=1)
    ax[i_axis].set_ylim(df_spindle_plot['q'].quantile(0.01), df_spindle_plot['q'].quantile(0.99))
    ax[i_axis].set_ylabel('SPq', labelpad=1, fontsize=fontsize_labels)
    # hlines at 0 and 0.25
    ax[i_axis].hlines([0], start_sec, end_sec, color='gray', alpha=0.3, lw=0.5)
    ax[i_axis].tick_params(axis='y', labelsize=fontsize_ticks)


    # _______________________________________________________________________________
    if 0: # plot time series signal overlayed with spindles and slow oscillations.
        i_axis += 1
        ax[i_axis].plot(data_plot['time'], data_plot['c4m1'], c='k', lw=1)
        ax[i_axis].set_ylim([-100, 100])
        ax[i_axis].set_ylabel('Amplitude (uV)')
        ax[i_axis].set_xlim(data_plot['time'].min(), data_plot['time'].max())

        for i, row in df_spindle_plot.iterrows():
            ax[i_axis].fill_between([row['start'], row['stop']], [-30, -30], [35, 35], color='r', alpha=0.2)

        ax[i_axis + 1].set_ylim(min(df_spindle_plot['q']), max(df_spindle_plot['q']))
        ax[i_axis + 1].set_ylabel('q')

        # plot mspindles ('msp_start' and 'msp_stop' columns):
        for i, row in df_mspindle_plot.iterrows():
            ax[i_axis].fill_between([row['msp_start'], row['msp_stop']], [35, 35], [40, 40], color='b', alpha=0.2)
            
        # plot SOs:
        for i, row in df_so_plot.iterrows():
            ax[i_axis].fill_between([row['start'], row['stop']], [-40, -40], [-32, -32], color='orange', alpha=0.2)

    # x ticks:
    ax[-1].set_xlim(data_plot['time'].min(), data_plot['time'].max())
    xticks = np.arange(start_sec, end_sec, 3600)
    xticklabels = np.arange(0, end_sec-start_sec, 3600) / 3600
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(xticklabels)
    ax[-1].set_xlabel('Time (h)')
    # ax[-1].set_xlabel('Time (s)')

    plt.subplots_adjust(hspace=0.01)

    plt.savefig(os.path.join(dir_figures, f'{study_id_savename}_figure_luna_1_spec_so_sp_fullnight.png'), dpi=300, bbox_inches='tight')
    
    
    
def plot_luna_so_detection_samples(data, df_so, dir_figures, n_so=25, study_id_savename=''):
    
    # similar to above, make a 10x1 figure with the following: select 10 SOs and plot them for a 20-second window. the 10 SOs should be chosen based on equidistant spacing in df_so 
    # (i.e. select 10 SOs that are equally spaced in time).

    # plot 10 SOs:
    so_spacing = int(df_so.shape[0] / n_so)

    df_so_plot = df_so.iloc[::so_spacing, :]
    df_so_plot = df_so_plot.iloc[:n_so, :]
    df_so_plot.reset_index(drop=True, inplace=True)

    fontsize_labels = 8
    fontsize_ticks = 6

    fig, ax = plt.subplots(n_so, 1, figsize=(10, n_so + 1), sharex=False)

    for i, row in df_so_plot.iterrows():
        
        dur_window = 20  # seconds
        start = row['start'] - dur_window / 2 - 0.5 
        stop = start + dur_window
        stage_tmp = data.loc[data['time'] >= row['start']].iloc[0]['stage']
        stage_tmp = stage_name_dict[stage_tmp]
        
        eeg_data_tmp = data.loc[(data['time'] >= start) & (data['time'] <= stop)]
        
        # for all SOs that are between start and stop, plot them:
        # ax[i_axis].fill_between([row['start'], row['stop']], [-30, -30], [35, 35], color='r', alpha=0.2)
        so_selection = df_so.loc[(df_so['start'] >= start-1) & (df_so['stop'] <= stop+1)]
        for j, row_so_in_window in so_selection.iterrows():
            ax[i].fill_between([row_so_in_window['start'], row_so_in_window['stop']], [-30, -30], [35, 35], color='orange', alpha=0.2)
        
        ax[i].plot(eeg_data_tmp['time'], eeg_data_tmp['c4m1'], c='k', lw=1)
        ax[i].set_ylim([-60, 75])
        ax[i].set_yticks([-30, 0, 35])
        ax[i].set_xlim(start, stop)
        # y tick fontsizes:
        ax[i].tick_params(axis='y', labelsize=fontsize_ticks)
        start_minute = np.round(start / 60, 1)
        # make y label start_minute
        ax[i].set_ylabel('M ' + str(start_minute) + f'\n{stage_tmp}', fontsize=fontsize_labels)
        # no ticks, no tick labels:
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])


    # for ax[-1], make 1-second ticks with length 1, with no labels:
    ax[-1].set_xticks(np.arange(start, stop, 1))
    ax[-1].tick_params(axis='x', which='major', length=2, labelsize=fontsize_ticks)

    ax[-1].set_xlabel('Time (s)', fontsize=fontsize_labels)
    # make an overall y-axis label for the figure:
    fig.text(0.01, 0.5, 'Amplitude (uV)', va='center', rotation='vertical', fontsize=fontsize_labels)

    plt.subplots_adjust(hspace=0)

    plt.savefig(os.path.join(dir_figures, f'{study_id_savename}_figure_luna_2_so_samples.png'), dpi=300, bbox_inches='tight')
    
    
    
def plot_luna_spindle_detection_samples(data, df_spindle, dir_figures, n_spindle=25, study_id_savename=''):
    
    # similar to above, make a 10x1 figure with the following: select spindles and plot them for a 20-second window. the 10 spindles should be chosen based on equidistant spacing in df_spindle 
    # (i.e. select 10 spindles that are equally spaced in time).
    

    spindle_spacing = int(df_spindle.shape[0] / n_spindle)
    spindle_spacing

    df_spindle_plot = df_spindle.iloc[::spindle_spacing, :]
    df_spindle_plot = df_spindle_plot.iloc[:n_spindle, :]
    df_spindle_plot.reset_index(drop=True, inplace=True)
    df_spindle_plot

    fontsize_labels = 8
    fontsize_ticks = 6

    fig, ax = plt.subplots(n_spindle, 1, figsize=(10, n_spindle + 1), sharex=False)

    for i, row in df_spindle_plot.iterrows():
        
        dur_window = 20  # seconds
        start = row['start'] - dur_window / 2 - 0.5 
        stop = start + dur_window
        stage_tmp = data.loc[data['time'] >= row['start']].iloc[0]['stage']
        stage_tmp = stage_name_dict[stage_tmp]
        q_tmp = np.round(row['q'], 2)
        
        eeg_data_tmp = data.loc[(data['time'] >= start) & (data['time'] <= stop)]
        
        # for all SOs that are between start and stop, plot them:
        # ax[i_axis].fill_between([row['start'], row['stop']], [-30, -30], [35, 35], color='r', alpha=0.2)
        spindle_selection = df_spindle.loc[(df_spindle['start'] >= start-1) & (df_spindle['stop'] <= stop+1)]
        for j, row_spindle_in_window in spindle_selection.iterrows():
            ax[i].fill_between([row_spindle_in_window['start'], row_spindle_in_window['stop']], [-30, -30], [35, 35], color='r', alpha=0.2)
        
        ax[i].plot(eeg_data_tmp['time'], eeg_data_tmp['c4m1'], c='k', lw=1)
        ax[i].set_ylim([-60, 75])
        ax[i].set_yticks([-30, 0, 35])
        ax[i].set_xlim(start, stop)
        # y tick fontsizes:
        ax[i].tick_params(axis='y', labelsize=fontsize_ticks)
        start_minute = np.round(start / 60, 1)
        # make y label start_minute
        # ax[i].set_ylabel('M' + str(start_minute) + f' ({stage_tmp})\nq={q_tmp}', fontsize=fontsize_labels)
        ax[i].set_ylabel('M ' + str(start_minute) + f' \n{stage_tmp}', fontsize=fontsize_labels)
        
        # no ticks, no tick labels:
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])
        # make text with "q" value in top center of plot:
        ax[i].text(start + dur_window / 2, 55, 'q='+str(q_tmp), ha='center', fontsize=fontsize_labels)
        
    # for ax[-1], make 1-second ticks with length 1, with no labels:
    ax[-1].set_xticks(np.arange(start, stop, 1))
    ax[-1].tick_params(axis='x', which='major', length=2, labelsize=fontsize_ticks)

    ax[-1].set_xlabel('Time (s)', fontsize=fontsize_labels)
    # make an overall y-axis label for the figure:
    fig.text(0.01, 0.5, 'Amplitude (uV)', va='center', rotation='vertical', fontsize=fontsize_labels)

    plt.subplots_adjust(hspace=0)

    plt.savefig(os.path.join(dir_figures, f'{study_id_savename}_figure_luna_3_spindle_samples.png'), dpi=300, bbox_inches='tight')
        


def luna_report_pdf(dir_figures, path_output_name=None, study_id_savename=None):
    """
    Pdf report of all full-night spec/so/sp and SO and SP samples.
    """
    if path_output_name is None:
        path_output_name = 'luna_report.pdf'
    # filepaths
    fp_in = f'{dir_figures}/*.png'

    png_files = glob.glob(fp_in)
    if study_id_savename is not None:
        png_files = [f for f in png_files if study_id_savename in f]
    png_files.sort()
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#pdf
    img, *imgs = [Image.open(f).convert('RGB') for f in png_files]
    path_output_name = os.path.join(dir_figures, f'{study_id_savename}_{path_output_name}')
    
    img.save(fp=path_output_name, format='PDF', resolution=100.0, save_all=True, append_images=imgs)
    
    
def load_luna_output_data(dir_results, verbose=False):
    """
    Load LUNA output data. All output files are expected to be in the same directory.
    """
    spindles_bl = pd.read_csv(os.path.join(dir_results, 'spindles_bl.csv'), index_col=0)
    spindles_f = pd.read_csv(os.path.join(dir_results, 'spindles_f.csv'), index_col=0)
    spindles_mspindle = pd.read_csv(os.path.join(dir_results, 'spindles_mspindle.csv'), index_col=0)
    spindles_ch_f = pd.read_csv(os.path.join(dir_results, 'spindles_ch_f.csv'), index_col=0)
    spindles_ch_f_spindle = pd.read_csv(os.path.join(dir_results, 'spindles_ch_f_spindle.csv'), index_col=0)
    so_ch = pd.read_csv(os.path.join(dir_results, 'so_ch.csv'), index_col=0)
    so_ch_e = pd.read_csv(os.path.join(dir_results, 'so_ch_e.csv'), index_col=0)
    so_ch_n = pd.read_csv(os.path.join(dir_results, 'so_ch_n.csv'), index_col=0)

    # for all of these dfs, try every column to set to float:
    for df in [spindles_ch_f, spindles_ch_f_spindle, spindles_mspindle, spindles_bl, spindles_f,
                so_ch, so_ch_e, so_ch_n]:
        for col in df.columns:
            col_new = col.strip().lower()
            df.rename(columns={col: col_new}, inplace=True)
            try:
                df[col_new] = df[col_new].astype(float)
            except ValueError:
                pass
            
            
    if verbose:
        n_head = 3
        print("\nspindles_bl: ", spindles_bl.shape, "M-Spindle summary per subject")
        print(spindles_bl.head(n_head))
        print("\nspindles_f: ", spindles_f.shape, "M-Spindle density per subject stratified by frequency")
        print(spindles_f.head(n_head))
        print("\nspindles_mspindle: ", spindles_mspindle.shape, "M-Spindle list for all subjects")
        print(spindles_mspindle.head(n_head))
        print("\nspindles_ch_f: ", spindles_ch_f.shape, 'Spindle summary statistics per channel stratified by frequency')
        print(spindles_ch_f.head(n_head))
        print("\nspindles_ch_f_spindle: ", spindles_ch_f_spindle.shape, 'Per-Spindle list')
        print(spindles_ch_f_spindle.head(n_head))
        print("\nso_ch: ", so_ch.shape, "SO summary statistics per channel")
        print(so_ch.head(n_head))
        print("\nso_ch_e: ", so_ch_e.shape, "SO summary per epoch per channel")
        print(so_ch_e.head(n_head))
        print("\nso_ch_n: ", so_ch_n.shape, "SO list per epoch per channel")
        print(so_ch_n.head(n_head))

    return spindles_bl, spindles_f, spindles_mspindle, spindles_ch_f, spindles_ch_f_spindle, so_ch, so_ch_e, so_ch_n


def read_luna_edf_xml(dir_data, study_id, verbose=False):
    """
    Read edf and xml files as used in LUNA computation.
    """

    # read edf file:
    fname = os.path.join(dir_data, study_id + '.edf')
    eeg = read_raw_edf(fname, preload=True, verbose=False)
    data = eeg.get_data()
    data = data * 1e6 # convert to uV
    fs = eeg.info['sfreq']
    data = filter_data(data, fs, l_freq=0.25, h_freq=22)
    channels = [x.lower() for x in eeg.info['ch_names']]
    data = pd.DataFrame(data.T, columns=channels)

    # read xml file
    stage = read_from_xml(os.path.join(dir_data, study_id + '.xml'), fs=200, epoch_length=30)

    data['stage'] = stage
    data['time'] = np.arange(data.shape[0]) / fs

    if verbose:
        print('Channels', channels)
        print('Fs', fs)
        print('Data', data.shape)

    return data, fs, channels