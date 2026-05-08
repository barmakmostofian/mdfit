Files and file formats:

`feat_matrix.csv` is the feature matrix in csv format.<br>
Each row corresponds to one simulation and each column to one feature. The row names are ligand names and, since the same protein-ligand system may have been simulated multiple times, rows with identical names may exist. The column (feature) names should be descriptive as they will be read in and ranked by the MDFitML.

`obs_pic50.csv` contains the observed response values (here: pIC50 activity values).<br>
For each ligand, one value is provided. The ligand names must correspond to those in the feature matrix. 

If, for some reason, a ligand is present in one file but not in the other (e.g. its activity value exists but it has not been simulated or vice versa), MDFitML will exclude that ligand from analysis when loading and merging these two files. 
