import subprocess
import os

results_dir = './results'

r_code = """library(luna)
k<-ldb("./results/out.db")

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
""" % ((results_dir,)*10)


r_code_path = './save_luna_output_as_csv.R'
with open(r_code_path, 'w') as ff:
    ff.write(r_code)

subprocess.check_call(['Rscript', r_code_path])
