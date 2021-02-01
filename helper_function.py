import numpy as np
import matplotlib.pyplot as plt
from math import isnan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xg

# a function to find avaialbel density or porosity logs for missing DTSM or DTCO or GRS
def log_depth_finder(x1,x2,lasDic,magic_list):
    wellNo = 0
    for filename in lasDic:
        print("on well: {}".format(filename))
        df = lasDic[filename].df()
        maxdepth = np.max(df.index)
        mindepth = np.min(df.index)
        # allmeasures = list(df.columns)
        df = df.drop(columns=[col for col in df if col not in magic_list])
        # replace all nan values with -1
        df = df.fillna(-1)
        allmeasures = list(df.columns)
        allmeasures.remove('DTSM')
        count_df = pd.DataFrame()
        for i in range(len(allmeasures)):
            df['DTSM' + str(i+1)] = np.multiply(df[allmeasures[i]],df['DTSM']).tolist()
            count_df['DTSM' + str(i+1)] = [np.shape(df[df['DTSM' + str(i+1)] != 1])[0]]

        # then find the column with the maximum number of rows which has value other than 1
        col_name = count_df.idxmax(axis=1)[0]
        df2  = df [['DTSM',col_name]]
        df_train = df2[(df2['DTSM']>0) & (df2[col_name]>0)]
        df_pred = df2[(df2['DTSM']==-1) & (df2[col_name]<0)] # multiple values by -1

        # do simple regression DTSM = f(DTSM1)
        xgb_r = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10)
        xgb_r.fit(np.asarray(df_train[col_name]), np.asarray(df_train['DTSM']))
        pred = xgb_r.predict(test_x)
        rmse = np.sqrt(MSE(test_y, pred))


def model_calibration_testing(xlabels, ylabels,lasDic):
    x = []
    y = []
    for filename in lasDic:
        allmeasures = list(lasDic[filename].df().columns)
        if all([item in allmeasures for item in xlabels + ylabels]):
            for index, row in lasDic[filename].df().iterrows():
                inputs = []
                [inputs.append(row[allmeasures.index(i)]) for i in xlabels]
                outputs = []
                [outputs.append(row[allmeasures.index(i)]) for i in ylabels]
                if all([not isnan(item) for item in inputs + outputs]):
                    x.append(inputs)
                    y.append(outputs)
    x = np.asarray(x)
    y = np.asarray(y)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    xgb_r = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10)
    xgb_r.fit(train_x, train_y)
    pred = xgb_r.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    data_size = len(x)
    return rmse, data_size


def model_calibration_testing_nan(xlabels, ylabels,lasDic):
    x = []
    y = []
    for filename in lasDic:
        allmeasures = list(lasDic[filename].df().columns)
        if all([item in allmeasures for item in ylabels]):
            for index, row in lasDic[filename].df().iterrows():
                inputs = []
                for i in xlabels:
                    try:
                        inputs.append(row[allmeasures.index(i)])
                    except:
                        inputs.append(np.nan)
                outputs = []
                [outputs.append(row[allmeasures.index(i)]) for i in ylabels]
                if all([not isnan(item) for item in outputs]):
                    x.append(inputs)
                    y.append(outputs)
    x = np.asarray(x)
    y = np.asarray(y)

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
    xgb_r = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10)
    xgb_r.fit(train_x, train_y)
    pred = xgb_r.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    rmse
    data_size = len(x)
    return rmse, data_size


# counting number of different well-log type (curve)
def curve_count(lasDic):
    curveCount = {}
    for filename in lasDic:
        las = lasDic[filename]
        for item in las.curves.iteritems():
            if item[0] in curveCount.keys():
                curveCount[item[0]] += 1
            else:
                curveCount[item[0]] = 1
    return curveCount


def curve_count_vis(curveCount,fig_name,tsh):
    keys = []
    for key in curveCount:
        if curveCount[key] > tsh:
            keys.append(key)
    values = [curveCount[key] for key in keys]
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values)
    plt.xticks(rotation=90)
    plt.savefig(fig_name + ".png")
    plt.show()


def well_log_vis(lasDic,filename):
    # Example - one single LAS file
    # Here we extract the actual log data from the las file.
    # filename = 'cbe115c74a89_TGS.las'
    las = lasDic[filename]
    logs = las.df()
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 8), sharey=True)
    ax[0].plot(logs.GRD, logs.index, color='green')
    ax[1].plot(logs.DTCO, logs.index, color='black')
    ax[2].plot(logs.DTSM, logs.index, color='c')
    for i in range(len(ax)):
        ax[i].set_ylim(logs.index[0], logs.index[-1])
        ax[i].invert_yaxis()
        ax[i].grid()

    ax[0].set_xlabel("GRD")
    ax[0].set_xlim(logs.GRD.min(), logs.GRD.max())
    ax[0].set_ylabel("Depth(ft)")
    ax[1].set_xlabel("DTCO")
    ax[1].set_xlim(logs.DTCO.min(), logs.DTCO.max())
    ax[2].set_xlabel("DTSM")
    ax[2].set_xlim(logs.DTSM.min(), logs.DTSM.max())
    plt.savefig("one_sample_well-log.png")
    plt.show()



def predictive_model(xlabels, ylabels,lasDic,lasDic_test,filename_test):
    x = []
    y = []
    for filename in lasDic:
        allmeasures = list(lasDic[filename].df().columns)
        if all([item in allmeasures for item in xlabels + ylabels]):
            for index, row in lasDic[filename].df().iterrows():
                inputs = []
                [inputs.append(row[allmeasures.index(i)]) for i in xlabels]
                outputs = []
                [outputs.append(row[allmeasures.index(i)]) for i in ylabels]
                if all([not isnan(item) for item in inputs + outputs]):
                    x.append(inputs)
                    y.append(outputs)
    x = np.asarray(x)
    y = np.asarray(y)


    # test
    test_x = []
    test_DF = lasDic_test[filename_test].df()
    # Todo: replace fill nan with log_depth_finder function
    test_DF = test_DF.fillna(0)
    allmeasures = list(test_DF.columns)
    if all([item in allmeasures for item in xlabels]):
        for index, row in test_DF.iterrows():
            inputs = []
            [inputs.append(row[allmeasures.index(i)]) for i in xlabels]
            if all([not isnan(item) for item in inputs]):
                test_x.append(inputs)
    test_x = np.asarray(test_x)

    xgb_r = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10)
    xgb_r.fit(x, y)
    pred = xgb_r.predict(test_x)
    return pred
