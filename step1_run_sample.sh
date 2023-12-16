# Step 1: create sample.lst, e.g. with this simple script or manually:
# source /home/wolfgang/anaconda3/etc/profile.d/conda.sh
# conda activate oracle
# python create_sample_db.py


# Step 2: Run LUNA:

if [ ! -d "./results/" ]; then
    mkdir -p "./results/"
fi


luna ./data/sample.lst -o ./results/out.db -s 'MASK ifnot-any=N2,N3 & RE & SPINDLES sig=C4M1 fc-lower=10 fc-upper=16 fc-step=1 q=-2 min=0.25 max=3.5 merge=0.5 per-spindle collate & so sig=C4M1 uV-neg=-35 uV-p2p=70 f-lwr=0.3, f-upr=4 verbose'
python save_luna_output_as_csv.py

# to view in LunaR:
# R
# library(luna)
# k <- ldb('./out.db')
# print(k)






