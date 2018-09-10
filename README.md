# DeepPerf

# training twitter model

th train1.lua -data_file data/twit/twit-train.hdf5 -val_data_file data/twit/twit-val.hdf5 -savefile twit-model

# evaluate
th evaluate1.lua -model twit-model_final.t7 -src_file data/twit/src-val.txt -output_file pred.txt -src_dict data/twit/twit.src.dict -targ_dict data/twit/twit.targ.dict
