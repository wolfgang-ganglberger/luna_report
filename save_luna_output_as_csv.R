library(luna)
    k<-ldb("./luna_output/out.db")

    write.csv(lx(k, "MASK", "EMASK"), "./luna_output/mask_emask.csv", row.names = FALSE)
    write.csv(lx(k, "RE", "BL"), "./luna_output/re_bl.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "BL"), "./luna_output/spindles_bl.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "F"), "./luna_output/spindles_f.csv", row.names = FALSE)                         
    write.csv(lx(k, "SPINDLES", "MSPINDLE"), "./luna_output/spindles_mspindle.csv", row.names = FALSE)          # m-spindle data from collate
    write.csv(lx(k, "SPINDLES", "CH_F"), "./luna_output/spindles_ch_f.csv", row.names = FALSE)
    write.csv(lx(k, "SPINDLES", "CH_F_SPINDLE"), "./luna_output/spindles_ch_f_spindle.csv", row.names = FALSE)
    write.csv(lx(k, "so", "CH"), "./luna_output/so_ch.csv", row.names = FALSE)                                   # so command
    write.csv(lx(k, "so", "CH_E"), "./luna_output/so_ch_e.csv", row.names = FALSE)                               # so verbose command
    write.csv(lx(k, "so", "CH_N"), "./luna_output/so_ch_n.csv", row.names = FALSE)                               # so verbose command
    