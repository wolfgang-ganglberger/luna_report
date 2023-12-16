library(luna)
k<-ldb("./results/out.db")

write.csv(lx(k, "MASK", "EMASK"), "./results/mask_emask.csv", row.names = FALSE)
write.csv(lx(k, "RE", "BL"), "./results/re_bl.csv", row.names = FALSE)
write.csv(lx(k, "SPINDLES", "BL"), "./results/spindles_bl.csv", row.names = FALSE)
write.csv(lx(k, "SPINDLES", "F"), "./results/spindles_f.csv", row.names = FALSE)                         
write.csv(lx(k, "SPINDLES", "MSPINDLE"), "./results/spindles_mspindle.csv", row.names = FALSE)          # m-spindle data from collate
write.csv(lx(k, "SPINDLES", "CH_F"), "./results/spindles_ch_f.csv", row.names = FALSE)
write.csv(lx(k, "SPINDLES", "CH_F_SPINDLE"), "./results/spindles_ch_f_spindle.csv", row.names = FALSE)
write.csv(lx(k, "so", "CH"), "./results/so_ch.csv", row.names = FALSE)                                   # so command
write.csv(lx(k, "so", "CH_E"), "./results/so_ch_e.csv", row.names = FALSE)                               # so verbose command
write.csv(lx(k, "so", "CH_N"), "./results/so_ch_n.csv", row.names = FALSE)                               # so verbose command
