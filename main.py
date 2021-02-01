"""
Sonic log predictive model
missing data

By Siavash 02/01/2021

"""

import lasio
import os
import pandas as pd
from helper_function import model_calibration_testing, model_calibration_testing_nan
from helper_function import curve_count_vis, curve_count,predictive_model

Main_dir = os.getcwd()
Temp_dir = Main_dir + str('\\') + 'Data'
os.chdir(Temp_dir)

lasDic = {}
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".las"):
        lasDic[filename] = lasio.read(filename)

os.chdir(Main_dir)

##############################################################################
# proposed workflow for tis project
# y = DTSM
# X = DTCO, GRS, etc ??????????? (sensitivity analysis and knowledge-based)
# step 1: missing data points a long the wellbore: since DTSM can be used to
# estimate porosity, let's use porosity and density and similar logs to
# estimate missing data points in depth
# step 2: y = f(x), predictive model.

# Todo: taking care of missing data using well-logs itself
magic_list = ['DPHZ_LS','RHOZ','DTSM','DPHZ_LS','NPHI','DPHI','RHOB','RHOM'
              ,'DPHZ_LS','NPHI_LS','DPHI_LS','SPHI_LS','TNPHI','TNPHI_LS'
    ,'GRS','DTCO','GRD','TNPH_LS']

# make sure we have DTSM and DTCO in files
# log_depth_finder(x1,x2,lasDic,magic_list)
##############################################################################
# sensitivity analysis with respect to different features (Try and error) - quick
# Todo: Implement other machine learning algorithms
# Method 1
# proposed method by Babak
xlabels = ['GRD', 'DTCO', 'SPR', 'RHOB']
ylabels = ['DTSM']
rmse, data_size = model_calibration_testing(xlabels, ylabels,lasDic)


# Method 2 by Siavash
xlabels = ['GRS','GRD', 'SPR', 'RHOB','NPHI','DPHI']
ylabels = ['DTSM']
rmse2, data_size2 = model_calibration_testing(xlabels, ylabels,lasDic)


# Method 3 by Siavash
xlabels = ['GRS','GRD', 'SPR', 'RHOB','NPHI','DPHI']
ylabels = ['DTSM']
rmse3, data_size3 = model_calibration_testing_nan(xlabels, ylabels,lasDic)


# Method 4 by Babak
xlabels = ['GRD', 'DTCO', 'SPR', 'RHOB']
ylabels = ['DTSM']
rmse4, data_size4 = model_calibration_testing_nan(xlabels, ylabels,lasDic)


# Method 5 by Siavash
xlabels = ['GRD', 'DTCO', 'SPR', 'RHOB','DPHI_LS','NPHI_LS']
ylabels = ['DTSM']
rmse5, data_size5 = model_calibration_testing(xlabels, ylabels,lasDic)


# Method 6 by Siavash
xlabels = ['GRD', 'DTCO', 'SPR', 'RHOB','DPHI_LS','NPHI_LS']
ylabels = ['DTSM']
rmse6, data_size6 = model_calibration_testing_nan(xlabels, ylabels,lasDic)


# find the best model (lowest rmse) using training data set
RMSE = [rmse, rmse2, rmse3, rmse4,rmse5,rmse6]
Data_size = [data_size, data_size2,data_size3,data_size4,data_size5,data_size6]
d = {'RMSE': RMSE, 'Data_size': Data_size}
Output_DF = pd.DataFrame(data=d)
writer = pd.ExcelWriter('output_models.xlsx')
Output_DF.to_excel(writer, 'Sheet1')
writer.save()


# Studying test dataset
Temp_dir2 = Main_dir + str('\\') + 'Test_data'
os.chdir(Temp_dir2)

lasDic_test = {}
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".las"):
        lasDic_test[filename] = lasio.read(filename)

os.chdir(Main_dir)

curveCount = curve_count(lasDic_test)
curve_count_vis(curveCount,fig_name = 'test_logs',tsh=4)


# use the entire data set as a training data set and given 10 wells as a test data set
# predictive model
xlabels = ['GRD', 'DTCO', 'RHOB','DPHI_LS']
pred_well1 = predictive_model(xlabels, ylabels,lasDic,lasDic_test,'Well 2.las')