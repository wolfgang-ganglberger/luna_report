import subprocess
import argparse

def main(luna_output_dir = './luna_output'):
    """
    Save LUNA output as csv files.
    """

    r_code = """library(luna)
    k<-ldb("%s/out.db")

    write.csv(lx(k, "MASK", "EMASK"), "%s/mask_emask.csv", row.names = FALSE)
    write.csv(lx(k, "RE", "BL"), "%s/re_bl.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "BL"), "%s/spindles_bl.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "F"), "%s/spindles_f.csv", row.names = FALSE)                         
    write.csv(lx(k, "SPINDLES", "MSPINDLE"), "%s/spindles_mspindle.csv", row.names = FALSE)          # m-spindle data from collate
    write.csv(lx(k, "SPINDLES", "CH_F"), "%s/spindles_ch_f.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "CH_F_SPINDLE"), "%s/spindles_ch_f_spindle.csv", row.names = FALSE)
    write.csv(lx(k, "so", "CH"), "%s/so_ch.csv", row.names = FALSE)                                   # so command
    write.csv(lx(k, "so", "CH_E"), "%s/so_ch_e.csv", row.names = FALSE)                               # so verbose command
    write.csv(lx(k, "so", "CH_N"), "%s/so_ch_n.csv", row.names = FALSE)                               # so verbose command
    """ % ((luna_output_dir,)*11)


    r_code_path = './save_luna_output_as_csv.R'
    with open(r_code_path, 'w') as ff:
        ff.write(r_code)

    subprocess.check_call(['Rscript', r_code_path])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Save LUNA output as csv files.')
    parser.add_argument('--luna_output_dir', type=str, default='./luna_output', help='directory where the LUNA output files are stored')
    
    args = parser.parse_args()

    main(luna_output_dir=args.luna_output_dir)