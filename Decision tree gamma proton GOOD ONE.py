# import relevant libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier 
from xgboost import plot_importance

# using XG boost because we have a lot of null values and it handles null values better than scikit-learn

#import the data files

df_gamma = pd.read_csv('gamma (1).csv')
df_proton = pd.read_csv('proton (1).csv')


#We are interested at only the records that has rec at the beginning
#Lets print them
for item in df_gamma.columns:
    if(item[:3]=='rec'):
        print(item)

# -Now I am going to remove records that are not interested-
# * rec.status/U/1 - This is saying the status of the simulation
# * rec.version/U/1 - This is just the version 
# * rec.eventID/U/1 - This is an ID
# * rec.runID/U/1 - This is an ID
# * rec.timeSliceID/U/1 - This is an ID
# * rec.trigger_flags/U/1 - This just a flag
# * rec.event_flags/U/1 - This is just a flag
# * rec.gtc_flags/U/1 - this is just a flag
# * rec.gpsSec/U/1 - Observed time is not an interest
# * rec.gpsNanosec/U/1 - Observed time is not an interest
# * rec_angleFitStatus_U_1 - This is a status
# * rec_planeNDOF_U_1
# * rec_SFCFNDOF_U_1
# * rec_coreFitStatus_U_1
# * rec_coreFiduScale_U_1
# * rec_LHLatDistFitNHitTanksMA_U_1
# * rec_LHLatDistFitNHitTanksOR_U_1
# * rec_LHLatDistFitNGoodTanksMA_U_1
# * rec_LHLatDistFitNZeroTanksMA_U_1
# * rec_LHLatDistFitNZeroTanksOR_U_1
# 
# 
# * rec_dec_F_0.0001 -- This is an importnat point that we know G/H seperation should not depend on dec
# * rec_ra_F_0.0001 -- This is an importnat point that we know G/H seperation should not depend on ra
# 
# I do not know about the following
# * rec_protonlheEnergy_F_0.01
# * rec_protonlheLLH_F_0.01
# * rec_gammalheEnergy_F_0.01
# * rec_gammalheLLH_F_0.01
# * rec_chargeFiduScale50_F_0.01
# * rec_chargeFiduScale70_F_0.01
# * rec_chargeFiduScale90_F_0.01
# 
# * rec_LDFAge_F_0.01
# * rec_LDFAmp_F_0.01
# * rec_LDFChi2_F_0.01
# * rec_GamCoreAge_F_0.01
# * rec_GamCoreAmp_F_0.01
# * rec_GamCoreChi2_F_0.01
# * rec_GamCorePackInt_F_0.01
# * rec_mPFnHits_F_0
# * rec_mPFnPlanes_F_0
# * rec_mPFp0nAssign_F_0
# * rec_mPFp0Weight_F_0.01
# * rec_mPFp0toangleFit_F_0.001
# * rec_mPFp1nAssign_F_0
# * rec_mPFp1Weight_F_0.01
# * rec_mPFp1toangleFit_F_0.001
# 
# * rec_disMax_F_0.01
# * rec_TankLHR_F_0.01
# * rec_LHLatDistFitXmax_F_0.01
# * rec_LHLatDistFitEnergy_F_1e-06
# * rec_LHLatDistFitMinLikelihood_F_0.01
# * rec_LHLatDistFitGoF_F_0.01
# * rec_LHXcog_F_0.01
# * rec_LHYcog_F_0.01
# * rec_LHLatDistFitZeroMinLikelihood_F_0.01
# * rec_LHLatDistFitHitMinLikelihood_F_0.01
# 
# Following is the list that I selected to keep
# 
# * rec_nChTot_U_1
# * rec_nChAvail_U_1
# * rec_nHitTot_U_1
# * rec_nHit_U_1
# * rec_nHitSP10_U_1
# * rec_nHitSP20_U_1
# * rec_nTankTot_U_1
# * rec_nTankAvail_U_1
# * rec_nTankHitTot_U_1
# * rec_nTankHit_U_1
# * rec_windowHits_U_1
# 
# * rec_CxPE40PMT_U_1
# * rec_CxPE40XnCh_U_1
# * rec_zenithAngle_F_0.0001
# * rec_azimuthAngle_F_0.0001
# 
# * rec_planeChi2_F_0.01
# * rec_coreX_F_0.1
# * rec_coreY_F_0.1
# * rec_logCoreAmplitude_F_0.1
# * rec_coreFitUnc_F_0.1
# * rec_SFCFChi2_F_0.01
# * rec_logNNEnergy_F_0.01
# * rec_fAnnulusCharge0_F_0.01
# * rec_fAnnulusCharge1_F_0.01
# * rec_fAnnulusCharge2_F_0.01
# * rec_fAnnulusCharge3_F_0.01
# * rec_fAnnulusCharge4_F_0.01
# * rec_fAnnulusCharge5_F_0.01
# * rec_fAnnulusCharge6_F_0.01
# * rec_fAnnulusCharge7_F_0.01
# * rec_fAnnulusCharge8_F_0.01
# 
# * rec_logMaxPE_F_0.01
# * rec_logNPE_F_0.01
# * rec_CxPE40_F_0.01
# * rec_CxPE40SPTime_F_0.1
# 
# * rec_PINC_F_0.01
# 


df_gamma_selected = df_gamma[['rec_nChTot_U_1', 'rec_nChAvail_U_1', 'rec_nHitTot_U_1', 'rec_nHit_U_1', 'rec_nHitSP10_U_1', 'rec_nHitSP20_U_1',                              'rec_nTankTot_U_1', 'rec_nTankAvail_U_1', 'rec_nTankHitTot_U_1', 'rec_nTankHit_U_1', 'rec_windowHits_U_1',                              'rec_CxPE40PMT_U_1', 'rec_CxPE40XnCh_U_1', 'rec_zenithAngle_F_0.0001', 'rec_azimuthAngle_F_0.0001', 'rec_planeChi2_F_0.01',                              'rec_coreX_F_0.1', 'rec_coreY_F_0.1', 'rec_logCoreAmplitude_F_0.1', 'rec_coreFitUnc_F_0.1', 'rec_SFCFChi2_F_0.01',                              'rec_logNNEnergy_F_0.01', 'rec_fAnnulusCharge0_F_0.01', 'rec_fAnnulusCharge1_F_0.01', 'rec_fAnnulusCharge2_F_0.01',                              'rec_fAnnulusCharge3_F_0.01', 'rec_fAnnulusCharge4_F_0.01', 'rec_fAnnulusCharge5_F_0.01', 'rec_fAnnulusCharge6_F_0.01',                              'rec_fAnnulusCharge7_F_0.01', 'rec_fAnnulusCharge8_F_0.01', 'rec_logMaxPE_F_0.01', 'rec_logNPE_F_0.01', 'rec_CxPE40_F_0.01',                              'rec_CxPE40SPTime_F_0.1', 'rec_PINC_F_0.01' ]].copy()
#Remove any raw with a nan
df_gamma_selected.dropna(inplace=True)
#Add a new column with the label 1- Gamma 0-Hadron
df_gamma_selected['label'] = 1

df_gamma_keeping_for_testing = df_gamma_selected[:5]


df_proton_selected = df_proton[['rec_nChTot_U_1', 'rec_nChAvail_U_1', 'rec_nHitTot_U_1', 'rec_nHit_U_1', 'rec_nHitSP10_U_1', 'rec_nHitSP20_U_1',                              'rec_nTankTot_U_1', 'rec_nTankAvail_U_1', 'rec_nTankHitTot_U_1', 'rec_nTankHit_U_1', 'rec_windowHits_U_1',                              'rec_CxPE40PMT_U_1', 'rec_CxPE40XnCh_U_1', 'rec_zenithAngle_F_0.0001', 'rec_azimuthAngle_F_0.0001', 'rec_planeChi2_F_0.01',                              'rec_coreX_F_0.1', 'rec_coreY_F_0.1', 'rec_logCoreAmplitude_F_0.1', 'rec_coreFitUnc_F_0.1', 'rec_SFCFChi2_F_0.01',                              'rec_logNNEnergy_F_0.01', 'rec_fAnnulusCharge0_F_0.01', 'rec_fAnnulusCharge1_F_0.01', 'rec_fAnnulusCharge2_F_0.01',                              'rec_fAnnulusCharge3_F_0.01', 'rec_fAnnulusCharge4_F_0.01', 'rec_fAnnulusCharge5_F_0.01', 'rec_fAnnulusCharge6_F_0.01',                              'rec_fAnnulusCharge7_F_0.01', 'rec_fAnnulusCharge8_F_0.01', 'rec_logMaxPE_F_0.01', 'rec_logNPE_F_0.01', 'rec_CxPE40_F_0.01',                              'rec_CxPE40SPTime_F_0.1', 'rec_PINC_F_0.01' ]].copy()
df_proton_selected.dropna(inplace=True)
df_proton_selected['label'] = 0

# this is optional at this level for this project. This doesn't get touched. But in the split part you are using the other part not dealt with here.
df_proton_keeping_for_testing = df_proton_selected[:5]


print(len(df_gamma_selected.index))
print(len(df_proton_selected.index))


#lets do some visualization
plt.hist(df_gamma_selected['rec_nHitTot_U_1'],label='Gamma',bins=100)
plt.hist(df_proton_selected['rec_nHitTot_U_1'],label='Proton',alpha=0.5,bins=100)
plt.legend()

#looking at the following we can say that rec_nHitTot_U_1 does not have a good classification power. 
#In other words it looks like this variable can not help us identifying whether the shower is gamma or proton
#At this point I am just keepting this information in my mind, but going to keep this variable


plt.hist(df_gamma_selected['rec_PINC_F_0.01'],label='Gamma',bins=100)
plt.hist(df_proton_selected['rec_PINC_F_0.01'],label='Proton',alpha=0.5,bins=100)
plt.legend()

#It looks like this variable has a real power seperating Protons from Gamma.


#Now combine gamma and proton
full_data = df_gamma_selected[5:].append(df_proton_selected[5:])
#Randomly shuffles the data
full_data = full_data.sample(frac=1)

# Splitting the data into train and test sets https://machinelearningmastery.com/evaluate-gradient-boosting-models-xgboost-python/

y_train = full_data['label']
full_data.drop(labels="label", axis=1, inplace=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(full_data)


state = 12  
test_size = 0.20  

# If you have 100 columns in original xtrain it'll split the columns for 70 columns in X train and 30 columns in y val
# Split into a 70:30 ratio between X train and X val. In the next line you train the xgboost classifier 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

np.shape(np.where(y_train==0)) # inserted to help with over training. This finds how many hadrons is in the dataset
np.shape(np.where(y_train==1)) # inserted to help with over training. This finds how many gamma is in the dataset. We want these to be roughly equal to make sure the data set isn't biased.
np.shape(y_train) # inserted to help with over training

np.shape(np.where(y_val==0)) # inserted to help with over training. Now doing it on the data we didn't feed the model. This finds how many hadrons is in the dataset
np.shape(np.where(y_val==1)) # inserted to help with over training. This finds how many gamma is in the dataset. We want these to be roughly equal to make sure the data set isn't biased.
np.shape(y_val) # inserted to help with over training. For both the datas it's pretty balanced. But the data is still overtrained.

# fit the model. Creating the decision tree here.
# training the xgboost classifier
xgb_clf = XGBClassifier(use_label_encoder=False) # create the object
xgb_clf.fit(X_train, y_train) # train the model. Took the training bucket and put that into the model and asked the model to train itself.

# evaluate the accuracy of the predictions
# Calculating the score using the validation test
score = xgb_clf.score(X_val, y_val) # there is another bucket of data which is the other 30% of the data set. This data wasn't shown to the model. Now in this line we are giving the data to the model and seeing what the model's accuracy is in splitting this data. Here the accuracy is only 80%. This is because of overtraining. With the training data we got 93% but when we give it independent data we only get 80%. This happens when we provide the model too much data to train on and the model gets too focused on the features of the training dataset. 
# we could fix the over training by reducing the training size. Originally test_size was 0.3, and we changed it to 0.5 to do this. But doing that made 79% and 99% accuracy, so that still didn't work because we want the accuracy of the training and the validation to be around the same number.
# changing around test size to find the best. Making it bigger made it worse.
score_train = xgb_clf.score(X_train, y_train) # finding the median accuracy the trained model was able to reach using the provided data set
print(score, score_train) # the values printed show how accurate the model is. The accuracy of the score is lower than that of score_train (what the trained dataset produced)
# this means that there are some features in the data, and we ask the model to classify the data between the gamma and hadron. The model has an accuracy of 93%, so its accuracy in classifying gamma/hadron data is 93%
# another step to fix the over training is to find the features and drop the features we don't care about to reduce the size of the data set.
# so below make a histogram to show the most important features. This article helps understand feature selection: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/

plot_importance(xgb_clf)
plt.show()
feature_import = np.sort(xgb_clf.feature_importances_)
print(feature_import)

# select features using threshold
selection = SelectFromModel(xgb_clf, threshold=0.02, prefit=True)
select_X_train = selection.transform(X_train)
select_X_val = selection.transform(X_val)
np.shape(select_X_train)
# train model
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
# calculate the score 
score_selected = selection_model.score(select_X_train, y_train)
print(score_selected)
score_selected_val = selection_model.score(select_X_val, y_val)
print(score_selected_val)

# eval model
select_X_test = selection.transform(X_train)
y_pred = selection_model.predict(select_X_train)

# after this, the next step would be to do the hyperparameter tuning. The xgboost creates multiples trees (an ensemble of trees), and we need to combine these trees. The hyperparameter looks at the max and min number of trees we can create, looks at how many branches each tree could have (how many times does it split?). We want to optimize that. We could do the overtraining again with hyperparameters.
# When we talk about gammas we could look at how accurately it accepts gammas and rejects hadrons. We could look at what fraction of gammas are incorrectly rejected (a type 2 error). 
# Could look at area under the curve, learning rate.
