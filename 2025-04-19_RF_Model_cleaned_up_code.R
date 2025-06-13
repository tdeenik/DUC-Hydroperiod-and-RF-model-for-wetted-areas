#2025-04-19 cleaning up RF code that was used for DUC analysis of OW areas in Cariboo-Chilcotin
library(raster)
library(sf)
library(dplyr)
library(stringr)
library(tidyr)
library(janitor)
library(randomForest)
library(caret)
library(tidyverse)
library(ranger)
library(rsample)
library(corrplot)
library(sp)
library(vip)
library(pdp)
library(ggplot2)
library(mlr3)
library(mlr3spatiotempcv)
library(mlr3learners)
library(mlr3tuning)
library(mlr3verse)
library(whitebox)
library(spatstat)
library(leaflet)
library(mapview)
library(terra)

#Move all parameters into one folder then select it here:
tif_folder <- "G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/Sentinel2_DUC_S2_Data"

#2023 to start
# List TIFF files in the folder that contain "2023" in the filename
tif_files_2023 <- list.files(tif_folder, pattern = "2023_08_28.*\\.tif$", full.names = TRUE)

param <- stack(tif_files_2023)
print(names(param))

# Rename layers
names(param) <- c("NDMI_S2_2023_08_28.tif", "NDVI_S2_2023_08_28.tif", "NDWI_S2_2023_08_28.tif",
                  "S2_2023_08_28_Ba.tif", "S2_2023_08_28_Bb.tif", "S2_2023_08_28_Bc.tif", "S2_2023_08_28_Bd.tif", "S2_2023_08_28_Be.tif")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>>>STEP 2: Training points<<<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pts <- read_sf("G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/Wetland_100MileHouse_Analysis/Water_training_points/Classes_OW_Barren_Veg_pts.shp")

#match pts to param
param_df <- raster::extract(param, pts, df=TRUE)
param_out <- cbind(pts, param_df)
head(param_out) #looks good

#drop ID
param_out <- param_out[,-(2)]

#save dataframe: 
write.csv(param_out,"G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/R-Code/RF_Code_BU_and_Outputs/param_out_2025-04-19.csv", row.names=FALSE)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Step 4: Train Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(12345)
index_wet <- createDataPartition(param_out$Class, p = .8, 
                                 list = FALSE, 
                                 times = 1)

train_wet <- param_out[ index_wet,]
test_wet  <- param_out[-index_wet,]

#this should preserve the proportions of each wetland class. Check:
prop.table(table(param_out$Class))
#Barren         OW        Veg 
#0.03337612 0.93709884 0.02952503 <- not great! obviously

indat <- train_wet

indat %>% group_by(Class) %>% count()
#Class          n
#  1 Barren        22
#  2 OW           612
#  3 Veg          21

data_sf <- st_as_sf(indat)

#wetland response as factor
data_sf$Class <- as.factor(data_sf$Class)

#get ranger classification learner
mlr_learners$get("classif.ranger")

#set up the classification task
# sf
task <- as_task_classif_st(data_sf, target = "Class", backend = data_sf)

# Verify the result
print(task)

#create the ranger classification learner
learner = mlr3::lrn("classif.ranger", oob.error = TRUE,
                    importance = 'impurity', predict_type = 'prob')

#hyperparameter search space NOTE: mtry can not be larger than number of variables in data
search_space = ps(
  mtry = p_int(lower = 3, upper = 8), #I think I want to use all 3 param every time... changed this when I had 8 param
  num.trees = p_int(lower = 500, upper = 1000),
  sample.fraction = p_dbl(lower = 0.5, upper = 0.8),
  max.depth = p_int(lower = 20, upper = 100),
  min.node.size = p_int(lower = 20, upper = 100)
)

#create spatial resampling k-fold 10
resampling_sp = rsmp("repeated_spcv_coords", folds = 10, repeats = 1)

#classification error
#measure = msr("classif.ce") #this is not working for some reason

measure = msrs(c('classif.acc')) #tried this instead and it worked...

#terminate after 100 evaluations
evals100 = trm("evals", n_evals = 100)

#create the tuning instance
instance = TuningInstanceSingleCrit$new(
  task = task,
  learner = learner,
  resampling = resampling_sp,
  measure = measure,
  search_space = search_space,
  terminator = evals100
)

#create the tuner
tuner = tnr("grid_search", resolution = 5)

#optimize the hyperparameters THIS WILL TAKE A LONG TIME TO RUN #on this small area it only took 2mins
tuner$optimize(instance)

save.image('G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/Wetland_100MileHouse_Analysis/RF_Model/Saved_workspaces/optimize_tuner_2023-08-28_replacement.Rda')
#load("C:/Users/kdeenik/Documents/Thesis/R/optimize_tuner.RData")

#take the optimize hyperparameters and apply to the learner
learner$param_set$values = instance$result_learner_param_vals

#train the model using the selected hyperparameters THIS IS YOUR MODEL
learner$train(task)
learner

##~~~~~~~~~~~~~~~~~~~~~~#Running the RF model~~~~~~~~~~~~~~~~~~~~~~~~~~~

#identify hyperparameters that were selected
instance$result_learner_param_vals

#calculate variable importance
filter = flt("importance", learner = learner)
filter$calculate(task)

print(filter)

#this is the variable importance plot!
plot(filter)

#ONCE YOU GET TO HERE WE CAN RUN THE PREDICTION. NEEDS YOUR RASTER STACKS

RF_mod2 <- learner$train(task)

#param is still your stack of rasters
ALL_stacked <- param

#started running 2:20pm
#model_output1 <- predict(ALL_stacked, learner, filename="G:/CPCC/CLIMATE_CHANGE_ANALYSIS/Predictive_Model_For_Changing_Conditions/OUTPUTS/RF_Index1_Classpredict_2024-10-21_V2.tif", predict_type="prob", 
#                         index=1, na.rm=TRUE, progress="window", overwrite=FALSE)

#I want to run this one first since it is water
model_output2 <- predict(ALL_stacked, learner, filename="G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/Wetland_100MileHouse_Analysis/RF_Model/Outputs/RF_Classification_Water_2023-08-13.tif", predict_type="prob", 
                         index=2, na.rm=TRUE, progress="window", overwrite=FALSE)

#now stack each date and rerun the trained model on each date...
#go through each unique date, see below.
tif_files_2023 <- list.files(tif_folder, pattern = "2022_09_12.*\\.tif$", full.names = TRUE)
param <- stack(tif_files_2023)
# Rename layers - they all have to have these names for the model... oops
names(param) <- c("NDMI_S2_2023_08_28.tif", "NDVI_S2_2023_08_28.tif", "NDWI_S2_2023_08_28.tif",
                  "S2_2023_08_28_Ba.tif", "S2_2023_08_28_Bb.tif", "S2_2023_08_28_Bc.tif", "S2_2023_08_28_Bd.tif", "S2_2023_08_28_Be.tif")
ALL_stacked <- param

#Open Water
model_output2 <- predict(ALL_stacked, learner, 
                         filename="G:/Shared drives/ECCC  Ducks Unlimited - Cariboo and Chilcotin/Data/Wetland_100MileHouse_Analysis/RF_Model/Outputs/RF_Classification_Water_2022_09_12.tif", predict_type="prob", 
                         index=2, na.rm=TRUE, progress="window", overwrite=FALSE)



#model_output3 <- predict(ALL_stacked, learner, filename="G:/CPCC/CLIMATE_CHANGE_ANALYSIS/Predictive_Model_For_Changing_Conditions/OUTPUTS/RF_Index3_Classpredict_2024-10-21_V2.tif", predict_type="prob", 
#                         index=3, na.rm=TRUE, progress="window", overwrite=FALSE)


#DONE:2019_06_15, 2019_08_09, 2019_08_29, 2019_09_03,
#DONE:2020_07_14, 2020_07_29, 2020_09_07, 
#DONE:2021_06_29, #2021_07_04, #2021_07_09, #2021_07_14, #2021_07_19, #2021_07_29, #2021_08_03, #2021_08_13, 2021_09_02, 2021_09_07, 
#DONE:2022_08_08, #2022_08_18, #2022_09_02, 2022_09_07, 2022_09_12, 
#DONE: 2023_07_09, 2023_08_13, 2023_08_28, 2023_09_02, 
#DONE: 2024_07_08, #2024_07_13, #2024_08_02, 2024_08_07, 2024_09_01, 2024_09_06

#North Thompson and Middle Fraser
#2019 = normal
#2020 = normal
#2021 = drought level 3 and 4
#2022 = level 1 drought ... the fall became a worse drought
#2023 = level 3-5 drought
#2024 = level 1 and 2 drought


##~~~~~~~~~~~~~~~~~~~~~~~~~~~MODEL ACCURACY~~~~~~~~~~~~~~~~~~~~~~~~

#read in test data
#load('G:/R-Code/Workspace/test_wet_V2_09-05_08-29-2023.Rda')
indat_test <- test_wet

#drop rows with na
indat_test <- indat_test %>% drop_na()

data_sf_test <- st_as_sf(indat_test, coords = c('X', 'Y'), crs = 4326) #32611

#response used as factor
data_sf_test$Class <- as.factor(data_sf_test$Class)

PREDICT_NEW = learner$predict_newdata(newdata = data_sf_test, task = task)

#get scores
PREDICT_NEW$score
PREDICT_NEW

#pull out values from the predictclassif object
tab = as.data.table(PREDICT_NEW)
tab

#confusion matrix
con_mat = confusionMatrix(tab$truth,tab$response)
con_mat
# Confusion Matrix and Statistics
# 
# Reference
# Prediction Barren  OW Veg
# Barren      3   0   2
# OW          0 153   0
# Veg         0   0   5
# 
# Overall Statistics
# 
# Accuracy : 0.9877          
# 95% CI : (0.9564, 0.9985)
# No Information Rate : 0.9387          
# P-Value [Acc > NIR] : 0.002244        
# 
# Kappa : 0.8952          
# 
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: Barren Class: OW Class: Veg
# Sensitivity                1.00000    1.0000    0.71429
# Specificity                0.98750    1.0000    1.00000
# Pos Pred Value             0.60000    1.0000    1.00000
# Neg Pred Value             1.00000    1.0000    0.98734
# Prevalence                 0.01840    0.9387    0.04294
# Detection Rate             0.01840    0.9387    0.03067
# Detection Prevalence       0.03067    0.9387    0.03067
# Balanced Accuracy          0.99375    1.0000    0.85714
# 
# #######~~~~~~~~~~~~~~~~~~~~~~~~~~~

#i wonder how this accuracy would change on a different image scene, not the one it was trained on...

