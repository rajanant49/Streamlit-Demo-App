import streamlit as st
from PIL import Image
import cv2
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit example")

st.write("""
# Explore different classifier
Which one is the best ??
""")

dataset_name= st.sidebar.selectbox("Select Dataset",("IRIS","BreastCancer","WineDataset"))
st.write(dataset_name)

classifier_name=st.sidebar.selectbox("Select Classifier",("KNN","RandomForest","SVM"))

def get_dataset(dataset_name):
    if dataset_name=="IRIS":
        data=datasets.load_iris()
    elif dataset_name=="BreastCancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()

    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)

st.write("shape of the datasets",X.shape)
st.write("Number of classes",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params=dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        params["K"]=K

    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params['C']=C

    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators

    return params

params=add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=42)

    return clf

clf=get_classifier(classifier_name,params)

#Classification

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_pred,y_test)
st.write("Classifier = ",classifier_name)
st.write("Accuracy = ",np.round(acc*100,2),"%")

pca=PCA(2)
X_projected=pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)
# def load_image(image_file):
# 	img = Image.open(image_file)
# 	return img
#
# image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
# if image_file is not None:
#     file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
#     st.write(file_details)
#
#     img = load_image(image_file)
#     st.image(img,width=250,height=250)
#     image_array=np.asarray(img)
#     st.image(image_array,width=100,height=100)
