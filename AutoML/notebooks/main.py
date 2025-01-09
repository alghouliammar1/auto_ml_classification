#!/usr/bin/env python
# coding: utf-8




# get_ipython().run_line_magic('pip', 'install pyod')


#Main Libs
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
#models 
from pyod.models import lof
from pyod.models import mad
from pyod.models import iforest
#visulaizing libs
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def get_data(data_url):
    orig_url = data_url
    file_id = orig_url.split('/')[-2]
    data_path='https://drive.google.com/uc?export=download&id=' + file_id
    df = pd.read_csv(data_path)
    print(f'machine_temperature_system_failure.csv : {df.shape}')
    df.head(10)
    return df

def adding_anomly_points_data(df,anomaly_points):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #is anomaly? : True => 1, False => 0
    # by default nothing is an anomaly
    df['anomaly'] = 0
    # convert time slices of data as anomalies (in the anomaly column) by setting it to 1
    for start, end in anomaly_points:
        df.loc[((df['timestamp'] >= start) & (df['timestamp'] <= end)), 'anomaly'] = 1
    # print how many of each type do we have
    df['anomaly'].value_counts()
    return   df[['timestamp', 'value']], df['anomaly']
    
#this  should be move to run func 



# # 5. Modeling
# >We will build several anomaly detection models and compare them each other. Let's create our datset `X` and ground truth labels `y`. Remember to use `y` only to evaluate model performance. In reality you are not supposed to know `y` beforehand. So don't use it for modeling
# 
# We will train and evaluate the following models
# 
# - 3-Sigma
# - Boxplot
# - Local Outlier Factor (LOF)
# - Isolation Forest
# - Mean Absolute Deviation (MAD)

# ## Prepare Dataset

# In[87]:



def prepare_dataset(df):
    X = df[['timestamp', 'value']]
    y = df['anomaly']
    return X,y   
    




# ## Model: 1. The 3-sigma Model
# 
# Compute mean and standard deviation and use the 3-STD (3-sigma) rule to compute the lower and upper limit.
# 
# Remember,
# 
# LL = mean - 3sigma
# UL = mean + 3sigma
# 
# Find out if each row of data is an outlier or not based on the temperature (`value`) column from `X` and compute performance using the ground truth labels in `y` using `classification_report`

# ### Build Model and Compute Outliers

# In[88]:

def sigma3_model(X,y):
    mean = X['value'].mean()
    sigma = X['value'].std()

    UL = mean + 3*sigma
    LL = mean - 3*sigma
    #LL, UL # we can print this variables
    outliers = [1 if ((value < LL) or (value > UL)) else 0 for value in X['value']]
    #just for loging things 
    sigma3_model_report=classification_report(y, outliers,output_dict=True)
    print(sigma3_model_report)
    return sigma3_model_report,outliers

# this code can be passed to one function 
  # px.scatter(visual_df, x='timestamp', y='value', color=outliers, width=1558, height=737, color_continuous_scale='tealrose').update_traces(marker=dict(size=2)).show()


# ## Model: 2. The Box-Plot Model
# 
# Compute Q1 and Q3 and use the IQR rule to compute the lower and upper whiskers.
# 
# Remember,
# 
# LW = Q1 - 1.5xIQR
# UW = Q3 + 1.5xIQR
# 
# Find out if each row of data is an outlier or not based on the temperature (`value`) column from `X` and compute performance using the ground truth labels in `y` using `classification_report`

# ### Build Model and Compute Outliers

# In[93]:
def box_plot_model(X,y):
    Q1 = X['value'].quantile(0.25)
    Q3 = X['value'].quantile(0.75)
    IQR = Q3 - Q1
    UW = Q3 + (1.5*IQR)
    LW = Q1 - (1.5*IQR)
   # LW, UW we can log them 
    outliers = [1 if ((value < LW) or (value > UW)) else 0 for value in X['value']]
    box_plot_model_report=classification_report(y, outliers,output_dict=True)
    print(box_plot_model_report)# we can log them
    return box_plot_model_report,outliers

# ### Visualize Outliers using TimeSeries
# 
# Use a similar scatterplot as before

# In[96]:
# Visulazing the result of each model
def Visulaizing_models_outliears(outliers,visual_df):
    px.scatter(visual_df, x='timestamp', y='value', color=outliers, width=1558, height=737, color_continuous_scale='tealrose').update_traces(marker=dict(size=2)).show()


# ## Model: 3. The Local Outlier Factor Model
# 
# You have already used LOF and learnt about it before. Now use it for outlier detection and evaluate its performance and visualize the outliers.
# 
# Use a default contamination rate of 0.1 for this model
# 
# You can use the `pyod` library for this model

# ### Build Model and Compute Outliers

def local_outlier_factor_model(X,y):
    lof_model = lof.LOF(contamination=0.1)
    lof_model.fit(X[['value']])
    outliers = lof_model.predict(X[['value']])
    local_outlier_factor_model_report=(classification_report(y, outliers,output_dict=True))
    print(local_outlier_factor_model_report)
    return local_outlier_factor_model_report



# ## Model: 4. The Isolation Forest Model
# 
# You have already used IForest and learnt about it before. Now use it for outlier detection and evaluate its performance and visualize the outliers.
# 
# Use a default contamination rate of 0.1 for this model
# 
# You can use the `pyod` library for this model

# ### Build Model and Compute Outliers

# In[101]:


def isolation_forest_model(X,y):
    if_model = iforest.IForest(contamination=0.1)
    if_model.fit(X[['value']])

    outliers = if_model.predict(X[['value']])
    isolation_forest_model_report=(classification_report(y, outliers,output_dict=True))
    print(isolation_forest_model_report)
    return isolation_forest_model_report,outliers


# ## Model: 5. The Median Absolute deviation (MAD) Model
# 
# Median Absolute deviation (MAD) is usually used for univariate data. It is a statistical model and is a very simple measure of variation in a sample. In that sense, it is quite similar to the standard deviation in terms of measuring statistical dispersion.
# 
# For a univariate data set $X_1, X_2, ..., X_n$, the MAD is defined as the median of the absolute deviations (residuals) from the data's median $\tilde{X} = median(X)$:
# 
# $$MAD = median(|X_i - \tilde{X}|)$$
# 
# To calculate a range of values that will not be considered outliers, we take the median value of the data and add/substract the MAD value multiplied with a threshold multiplier $t$:
# 
# $$ \tilde{X} \pm MAD*t$$
# 
# 
# Luckily PyOD can do everything for us! You just need to specify the `threshold` value $t$ in the model
# 
# You can use the `pyod` library for this model
# 
# You can use the default threshold of 3.5.
# 
# 

# ### Build Model and Compute Outliers

# In[104]:



def MAD_model(X,y):
    mad_model = mad.MAD(3.5)
    mad_model.fit(X[['value']])
    outliers = mad_model.predict(X[['value']])
    MAD_model_report=classification_report(y, outliers,output_dict=True)
    print(MAD_model_report)
    return MAD_model_report,outliers


# ### Bonus: Rolling MAD
# For time-series data such as this we usually compute a "rolling MAD" for a window that is moving over the data and then have a series of median values and MAD thresholds. Calculate a "rolling MAD" and experiment with the window size.

# In[108]:


# import numpy as np
def rolling_MAD(df,windows=600):
    window = windows
    df_mad = df.copy(deep = True)
    mad = lambda x: np.median(np.fabs(x - x.median()))
    df_mad['rolling_median'] = df_mad['value'].rolling(window, center=True).median()
    df_mad['mad'] = df_mad['value'].rolling(window, center=True).apply(mad, raw = False)
    df_mad['upper'] = df_mad['rolling_median'] + df_mad['mad']*3.5
    df_mad['lower'] = df_mad['rolling_median'] - df_mad['mad']*3.5

    df_mad['anomaly'] = df_mad.apply((lambda x: 0 if (x['value'] <= x['upper']) and (x['value'] >= x['lower']) else 1), axis=1)

    rolling_MAD_report=(classification_report(df_mad['anomaly'], y,output_dict=True))
    print(rolling_MAD_report)
    return rolling_MAD_report,outliers

# Function to visualize classification reports
def visualize_classification_reports(model_reports):
    data = []
    for model_name, report in model_reports.items():
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Only process class metrics
                data.append({
                    "Model": model_name,
                    "Class": label,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1-score"]
                })

    df = pd.DataFrame(data)

    # Pivot the DataFrame for Heatmap
    heatmap_data = df.pivot(index="Class", columns="Model", values="F1-Score")


    # Plot Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues")
    plt.title("F1-Score Comparison Across Models and Classes")
    plt.ylabel("Class")
    plt.xlabel("Model")
    plt.show()
# def visualize_classification_reports(model_reports):
#     data = []
#     for model_name, report in model_reports.items():
#         for label, metrics in report.items():
#             if isinstance(metrics, dict):  # Only process class metrics
#                 data.append({
#                     "Model": model_name,
#                     "Class": label,
#                     "Precision": metrics["precision"],
#                     "Recall": metrics["recall"],
#                     "F1-Score": metrics["f1-score"]
#                 })

#     df = pd.DataFrame(data)

#     # Pivot the DataFrame for Heatmap
#     heatmap_data = df.pivot(index="Class", columns="Model", values="F1-Score")

#     # Plot Heatmap
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues")
#     plt.title("F1-Score Comparison Across Models and Classes")
#     plt.ylabel("Class")
#     plt.xlabel("Model")
#     plt.show()

   

def main():
   data= get_data("https://drive.google.com/file/d/12fFZ9k8wsmWBVUhcsVxmKsqHxaVzAzqt/view?usp=sharing")
   anomaly_points = [
        ["2013-12-10 06:25:00.000000","2013-12-12 05:35:00.000000"],
        ["2013-12-15 17:50:00.000000","2013-12-17 17:00:00.000000"],
        ["2014-01-27 14:20:00.000000","2014-01-29 13:30:00.000000"],
        ["2014-02-07 14:55:00.000000","2014-02-09 14:05:00.000000"]
    ]
   models = [
    "3-Sigma",
    "Boxplot",
    "Local Outlier Factor (LOF)",
    "Isolation Forest",
    "Mean Absolute Deviation (MAD)"
    ]
   X,y= adding_anomly_points_data(data,anomaly_points) 
   model_reports = {}

    # Run MAD model
   mad_report, mad_outliers = MAD_model(X, y)
   model_reports["MAD_Model"] = mad_report
    #isolation_forest_model
   isolation_forest_report, isolation_forest_outliers =isolation_forest_model(X, y)
   model_reports["isolation_forest"] = isolation_forest_report
    #visulaizing
   visualize_classification_reports(model_reports)
   
    
if __name__ == '__main__':
    main()