# DeepPerf


# For running DUPLE, DAME, DENIM and Struct-ANN

Go into the `deep_non_decomp_src` folder to see the code.

The following address are relative to the `deep_non_decomp_src` folder.

# Code for runnind DUPLE, DENIM, DAME and Struct ANN

I apologize in advance for this code being weirdly inconsistent in several ways. I have edited this code over a long period of time with significant breaks in between, which I blame for this inconsistency.

All the data is in the `datasets` folder and is read through the wrapper in `datasets/dataRead.py`.
## Concave measures and Benchmark

### To run DUPLE

1. Ensure that the variable `dual_class` in Line 15 is set to one of the classes in `DeeSpade.dual_step`.
2. Ensure that the variable `model` in Line 22 is set to `Spade`.

3. and then run

```python train_batch_opt.py [dataset]```

4. The score is accumulated in line 72 and 73.

5. Use lines 96 and 97 to save it to file.

### To run p-Benchmark (ANN-p)

1. The variable `dual_class` is inconsquential.
2. Ensure that the variable `model` in Line 22 is set to `BenchANN`.

3. and then run

```python train_batch_opt.py [dataset]```

4. All the scores are accumulated in `minC` in Line 71.
5. Save them through 98.

### To run Benchmark (ANN-0-1)

1. You will have to do a trivial change in the BenchANN file to get rid of the p-sensitive cost function to get the true cost. To do this comment Line 40 in DeeSpade/bench.py and uncomment Line 42```

2. The variable `dual_class` is inconsquential.
3. Ensure that the variable `model` in Line 22 is set to `BenchANN`.

3. and then run

```python train_batch_opt.py [dataset]```

4. All the scores are accumulated in `minC` in Line 71.
5. Comment Line 72 and Line 73.
6. Extract the different scores from Line 76 to 80.
7. Save them through Line 99 - 102.


## Pseudolinear Measures

### To run DAME
1. Fbeta score is the only score we see here. The code for that is in `DAMP.ANNAMP/FbetaANN`.
2. Run ```python ANNAMPTrain.py [dataset]```
3. The scores are stored in `[dataset]ANNAMAP_FMeas_new.npz`

## To run ANN-PG
1. The code is in `DAMP.AMP.FbetaThresh`.
2. Run ```python AMPTrain.py [dataset]```#
3. The scores are stored in `[dataset]AMP_PG.npz`

## Nested Concave Measures

1. Here we only look at NegKLD. The code is in `DAMP.AMP.FbetaThresh` and the primal and dual step are in `demesis.concave_fn.KLD`.
2. Run ```python train_denembis_kld.py [dataset]```
3. The score is stored in `[dataset]_kld_rew.npz`

``` Some files also calculate BAKLD but they can be ignored ```

## Struct ANN file

1. The `MVC` code is present in `all_struct/c_code/mvc.c` and the shared library is already compiled in the folder as `libmvc.so`.
2. This is then used by the network definition and training algorithm which is present in `all_struct/struct_ann.py` and the final training wrapper is `train_batch_struct.py`.
3. Ignoring the details, to train run the command ```python train_batch_struct.py [dataset] [loss_fn]```
where the `[dataset]` variable is as usual and the variable `[loss_fn]` is defined in `all_struct/loss_functions.py```. We only use `minTPRTNR` and `fone` among those.

## Plotting the File

1. Run the necessary training files to obtain the score files.
2. Then run the necessary plot file i.e one of
   * `plot_[Fmeas, KLD, MinTPRTNR, QMean].py [x_axis_length]`


# Twitter model

## Training the model

`th train1.lua -data_file data/twit/twit-train.hdf5 -val_data_file data/twit/twit-val.hdf5 -savefile twit-model`

## Evaluate
`th evaluate1.lua -model twit-model_final.t7 -src_file data/twit/src-val.txt -output_file pred.txt -src_dict data/twit/twit.src.dict -targ_dict data/twit/twit.targ.dict`


