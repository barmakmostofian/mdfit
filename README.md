# MDFitML

MDFitML is a framework for small-molecule potency prediction derived from protein-ligand simulation fingerprints (SimFPs). It is based on the [MDFit](https://dx.doi.org/10.1007/s10822-024-00564-2) workflow but it is significantly improved in terms of end-to-end automation and results interpretation. 

The basic strategy is to identify top-N (e.g. N=10) features that most contribute to separating strong from weak binders and to use these for potency prediction by ordinary least-squares regression. At the moment, the script provided performs only an L1-regularized regression on a feature matrix. This will soon be expanded by allowing the user to run L2-regularized, random-forest, or XGBoost regression models. However, our initial results indicate that the L1-regularized regression for feature elimination performs best (a manuscript is in preparation).

MDFitML achieves R<sup>2</sup> values comparable or superior to conventional alchemical binding free energy methods, particularly when taking transient interactions like water-bridged Hbonds into account. Of course, it is agnostic toward the MD simulation and analysis software, as long as the feature matrix follows the formatting guidelines (see `Data/`).  

The script can be run like this:<br>
`python run_mdfitml_lasso.py   --data_feat  merged_data.csv   --data_obs  obs_pic50.csv   --top_N 10`
