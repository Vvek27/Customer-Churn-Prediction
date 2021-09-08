from numpy.core.fromnumeric import var
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from imblearn.over_sampling import SMOTE
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")
import os
plt.style.use("ggplot")

# Dataframe
my_dataset="Churn_Modelling.csv"
def explore_data(dataset):
    df=pd.read_csv(dataset)
    return df
data=explore_data(my_dataset)
features=11

# target variable
target_var=data["Exited"]


# Data preprocessing
df=data.iloc[:,3:]
le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
a=pd.get_dummies(df["Geography"])
a=a.drop(a.columns[-1],axis=1)
df=pd.concat([df,a],axis=1)
df.drop("Geography",axis=1,inplace=True)
#x,y

X=df.drop("Exited",axis=1)
y=df["Exited"]

# balance the class
smote=SMOTE(sampling_strategy="minority")
X,y=smote.fit_resample(X,y)

#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Fature scalling
scaler=MinMaxScaler()
X_train[X_train.columns]=scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns]=scaler.transform(X_test[X_test.columns])
my_model=load_model("my_model.h5")


st.set_page_config(initial_sidebar_state="collapsed")

# Title
html_temp = """
    <div style="background-color:grey;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Churn Analysis</h2>
    </div>"""
st.markdown(html_temp,unsafe_allow_html=True)


# background
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://image.freepik.com/free-photo/gray-abstract-wireframe-technology-background_53876-101941.jpg")
 }
    </style>
    """,
    unsafe_allow_html=True
)

# sidebar
activities=["Select activity","EDA","Data Visualization","Prediction","Model statistics","About"]
choice=st.sidebar.selectbox("",activities)
if choice=="Select activity":
    activity()
if choice=="EDA":
    EDA()
if choice=="Data Visualization":
    Visualization()
if choice=="Prediction":
    predict()
if choice=="Model statistics":
    statistics()
if choice=="About":
    about()

def activity():
    def lottie_file(url:str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
    lottie_hello=lottie_file("https://assets6.lottiefiles.com/packages/lf20_kltum0us.json")
    st_lottie( lottie_hello, speed=1, reverse=False,loop=True,quality="low",
    renderer="svg")
def EDA():
    st.header("Exploratory Data Analysis")
    # Basic functions
    method_names=["Show dataset","Head","Tail","Shape","Describe","Missing value","Columns Names","Value counts"]
    method_operation=[data,data.head(),data.tail(),data.shape,data.describe(),data.isnull().sum(),data.columns,target_var.value_counts()]

    for i in range(len(method_names)):
        if st.checkbox(method_names[i]):
            st.write(method_operation[i])
    all_columns=list(data.columns)
    if st.checkbox("Select columns to show"):
        selected_columns=st.multiselect("Select column",all_columns)
        new_df=data[selected_columns]
        st.dataframe(new_df)

def Visualization():
    st.header("Data Visualization")
    # plots for numerical variable
    if st.checkbox("Numerical variable"):
        column_name=st.selectbox("",("Select column","CreditScore","Balance","EstimatedSalary","Age"))
        if column_name=="CreditScore":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
            st.write(sns.barplot(target_var,data[column_name]))
            st.pyplot()
            st.write(sns.boxplot(target_var,data[column_name]))
            st.pyplot()
        elif column_name=="Balance":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
            st.write(sns.barplot(target_var,data[column_name]))
            st.pyplot()
            st.write(sns.boxplot(target_var,data[column_name]))
            st.pyplot()
        elif column_name=="EstimatedSalary":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
            st.write(sns.barplot(target_var,data[column_name]))
            st.pyplot()
            st.write(sns.boxplot(target_var,data[column_name]))
            st.pyplot()
        elif column_name=="Age":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
            st.write(sns.barplot(target_var,data[column_name]))
            st.pyplot()
            st.write(sns.boxplot(target_var,data[column_name]))
            st.pyplot()
        # plots for categorical variable
    if st.checkbox("Categorical variable"):
        column_name=st.selectbox("",("Select column","Geography","Gender","Tenure","NumOfProducts"))
        if column_name=="Geography":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(pd.crosstab(data[column_name],target_var).plot(kind="bar"))
            st.pyplot()
        elif column_name=="Gender":
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(pd.crosstab(data[column_name],target_var).plot(kind="bar"))
            st.pyplot()
        elif column_name=="Tenure":
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(pd.crosstab(data[column_name],target_var).plot(kind="bar"))
            st.pyplot()
        elif column_name=="NumOfProducts":
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(pd.crosstab(data[column_name],target_var).plot(kind="bar"))
            st.pyplot()
    if st.checkbox("Target variable"):
        values=target_var.value_counts().values
        index=target_var.value_counts().index
        plt.figure(figsize=(6,4))
        st.write(target_var.value_counts().plot.pie(radius=0.75,autopct="%0.2F%%"))
        my_circle=plt.Circle( (0,0), 0.3, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        st.pyplot(fig=None)
        st.write(sns.barplot(index,values))
        st.pyplot()
        
    if st.checkbox("Correlation matrix"):
        plt.figure(figsize=(10,5))
        st.write(sns.heatmap(data.corr(),annot=True,cmap="Blues"))
        st.pyplot()

    if st.checkbox("Outiers analysis"):
        columns=["Balance","EstimatedSalary","Age","CreditScore"]
        for i in range(len(columns)):
            plt.figure(figsize=(10,4))
            plt.subplot(2,2,i+1)
            st.write(sns.boxplot(data[columns[i]]))
            st.pyplot()

def predict():
    col_list=[0]*features

    credit_col=st.number_input("Enter customer credit score",step=50)
    col_list[0]=credit_col

    gen_col=["Female","Male"]
    gen_num=list(range(len(gen_col)))
    gen=st.radio("Select gender",gen_num,format_func=lambda x:gen_col[x])
    col_list[1]=gen  

    age_col=st.slider("Enter the customer age",18,99)
    col_list[2]=age_col

    Tenure_col=st.slider("Enter tenure",0,10)
    col_list[3]=Tenure_col

    Balance_col=st.number_input("Enter customer balance",step=500)
    col_list[4]=Balance_col

    product=[1,2,3,4]
    product_option=st.selectbox("Select no. of products",product)
    for item in product:
        if item==product_option:
            col_list[5]=item
    
    credit_card=["No","Yes"]
    credit_num=list(range(len(gen_col)))
    credit=st.radio("Customer has credit card ?",credit_num,format_func=lambda x:credit_card[x])
    col_list[6]=credit

    active_member=["No","Yes"]
    member_num=list(range(len(gen_col)))
    active=st.radio("Customer is active member ?",member_num,format_func=lambda x:active_member[x])
    col_list[7]=active

    salary=st.number_input("Enter the salary of customer",step=1000)
    col_list[8]=salary

    geography=["Geography","France","Germany","Spain"]
    geography_option=st.selectbox("",geography)
    if geography_option=="France":
        col_list[9]=1
        col_list[10]=0
    elif geography_option=="Germany":
        col_list[9]=0
        col_list[10]=1
    elif geography_option=="Spain":
        col_list[9]=0
        col_list[10]=0

    if st.checkbox("Your entries"):
        d={}
        feature=["Credit score","Gender","Age","Tenure","Balance","Numofproducts","Hascrcard",
        "Active member","Salary","Geography"]
        for i in range(len(feature)):
            if i<9:
                d[feature[i]]=col_list[i]
            else:
                d[feature[i]]=[col_list[i],col_list[i+1]]
        st.write(d)

    if st.button("Predict"):
        predicted_value=my_model.predict(scaler.transform([col_list]))
        if predicted_value[0]<0.5:
            st.success("Happy! Customer is not leaving bank")
            st.write(0)
        else:
            st.warning("Customer is leaving the bank")
            st.write(1)
def statistics():
    y_test_pred=my_model.predict(X_test)
    y_test_pred=pd.DataFrame(y_test_pred)
    y_test_pred=round(y_test_pred)
    cf=confusion_matrix(y_test,y_test_pred)
    method_names=["Accuracy score","Confusion matrix","Classification report"]
    method_opt=[accuracy_score(y_test,y_test_pred),
    cf,classification_report(y_test,y_test_pred)]
    for i in range(len(method_names)):
        if st.button(method_names[i]):
            st.write(method_opt[i])
def about():
    st.text("streamlit app made by vivek patel")



























            









    
        

    
    














#    


    

    
