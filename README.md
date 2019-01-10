# julia-binary-classifiers
Classify gene data from R

Gleason Score correlates with cancer recurrance hence survival. Attempt to predict GS > 7 based on gene expression data. These cases were all men with prostrate cancer. Example data may not be public, although it is not possible to identify individuals.

Use R conv.R to convert bigmemory descriptors so Julia can load them (as lists).
Use julia ready_data.jl to prepare data by filtering missing or Gleason Score = 7

Use julia -p 8 -L select_training_subset.jl -e 'run();' to choose training subset (cols), and least correlated subset of features (rows).
Use julia top_v.jl to find (and then copy to all run*.h5) the best subset of features, and re-run the above to continue search.
There are 1.4 million features (genes), so a small subset (foldFeatures) are used to speed up the models.

Use julia -L predict_survival.jl then
- predict_survival.run(1) to use Linear (works well, even with all 1.4M features)
- predict_survival.run(2) to use SVM (bug unsafe_copyto! FIXED)
- predict_survival.run(3) to use DecisionTree (with copy because Library will not handle Adjoint)
- predict_survival.run(4) to use RandomForest (with copy because Library will not handle Adjoint)
- predict_survival.run(5) to use Booster (will not compile)
- predict_survival.run(6) to use Naive Bayes (slow, benefits from best subset of features)
- predict_survival.run(7) to use TensorFlow (bug with @tf macro reported; FIXED)
- predict_survival.run(8) to use Flux (will not run ERROR: TypeError: non-boolean (ForwardDiff.Dual{Nothing,Bool,1}) used in boolean context)
 
Will load saved state from run[1-8].h5 to determine training subset and best feature subset.
