import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from luna_post_analysis_fct import load_luna_output_data, read_luna_edf_xml, read_from_xml, plot_spectrogram_so_sp, plot_luna_so_detection_samples, plot_luna_spindle_detection_samples, luna_report_pdf

sys.path.append(os.path.expanduser('~/repos/sleep_cognition'))
from preprocessing_and_spectrograms import compute_eeg_spectrogram
    
# before running this script, you usually want to run LUNA on data from a to-compute-table
# and have the edf and xml and luna output files in the correct directories


def post_analysis_and_plotting(study_id, spindles_bl, spindles_f, spindles_mspindle, spindles_ch_f, spindles_ch_f_spindle, so_ch, so_ch_e, so_ch_n, dir_edf_xml_data, dir_figures):
    
    df_spindle = spindles_ch_f_spindle.loc[study_id]
    if 0:
        print(df_spindle['q'].describe())
    df_mspindle = spindles_mspindle.loc[study_id]
    df_so = so_ch_n.loc[study_id]

    data, fs, channels = read_luna_edf_xml(dir_edf_xml_data, study_id)

    eeg_channel = 'c4m1'

    # compute spectrogram
    specs, freq, specs_nopad = compute_eeg_spectrogram(data[eeg_channel].values, fmax_eeg=20, return_specs_wopad=True, hours_pad=9)

    # plot spectrogram with spindles and SOs detection (full night plot)
    plot_spectrogram_so_sp(specs_nopad, data, df_spindle, df_mspindle, df_so,
                                freq, dir_figures, study_id_savename=study_id)

    # plot samples of spindles and SOs detection
    plot_luna_so_detection_samples(data, df_so, dir_figures, n_so=25, study_id_savename=study_id)
    plot_luna_spindle_detection_samples(data, df_spindle, dir_figures, n_spindle=25, study_id_savename=study_id)

    # generate pdf report
    luna_report_pdf(dir_figures, study_id_savename=study_id)



def main(luna_output_dir='./luna_output', dir_edf_xml_data='./data', dir_figures = './figures', study_ids: list = None):
    """
    This script is used to generate figures for the LUNA output report.
    The figures are saved in the 'figures' directory.    
    """
        
    if not os.path.exists(dir_figures):
        os.mkdir(dir_figures)
        
    # load data for all subjects included in the luna output files:
    luna_output_data = load_luna_output_data(luna_output_dir)
    spindles_bl, spindles_f, spindles_mspindle, spindles_ch_f, spindles_ch_f_spindle, so_ch, so_ch_e, so_ch_n = luna_output_data

    # select a study id for which post-analysis and plotting should be performed:
    if study_ids is None:
        study_ids = spindles_ch_f_spindle.index.unique()

    for study_id in study_ids:
        print(study_id)
        post_analysis_and_plotting(study_id, spindles_bl, spindles_f, spindles_mspindle, spindles_ch_f, spindles_ch_f_spindle, so_ch, so_ch_e, so_ch_n, dir_edf_xml_data, dir_figures)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Save LUNA output as csv files.')
    parser.add_argument('--luna_output_dir', type=str, default='./luna_output', help='directory where the LUNA output files are stored')
    parser.add_argument('--dir_edf_xml_data', type=str, default='./data', help='directory where the edf and xml files are stored')
    parser.add_argument('--dir_figures', type=str, default='./figures', help='directory where the figures should be saved')
    parser.add_argument('--study_ids', type=str, nargs='+', default=None, help='list of study ids for which the figures should be generated')
    
    args = parser.parse_args()
    
    main(luna_output_dir=args.luna_output_dir, dir_edf_xml_data=args.dir_edf_xml_data, dir_figures=args.dir_figures, 
        study_ids=args.study_ids)