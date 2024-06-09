from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, recall_score
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import sklearn.metrics as mertics
from sklearn.metrics import classification_report
import os
from django.contrib import messages

def index(request):
    if os.path.exists('comp/model.pkl')==False:
        df=pd.read_csv("comp/cancer.csv")
        features = df.drop('Level',axis=1)
        features = features.drop('Patient Id',axis=1)
        features = features.drop('index',axis=1)
        features = features.drop('Gender',axis=1)
        Level=df["Level"]
        X_train,X_test,Y_train,Y_test = train_test_split(features,Level,test_size=0.20,random_state=0)
        y =df.Level
        X =df.drop("Level",axis=1)
        lr = LogisticRegression()
    
        lr.fit(X_train,Y_train)
        Y_pred_lr = lr.predict(X_test)
        score_lr =round(accuracy_score(Y_pred_lr,Y_test)*100,2)
        print("Accuracy using Logistic Regression: " + str(score_lr) + "%")
        with open('comp/model.pkl', 'wb') as file:
            pickle.dump(lr, file)
    else:
        pass
    prediction(request)
    return render(request, 'comp/can.html')

def prediction(request):
    if request.method == 'POST':
        pred_lr = ['None']
        df1 = pd.read_csv('comp/cancer.csv')
        features = df1.drop('Level',axis=1)
        features = features.drop('Patient Id',axis=1)
        features = features.drop('index',axis=1)
        features = features.drop('Gender',axis=1)
        with open('comp/model.pkl', 'rb') as file:
            lr = pickle.load(file)
        data = {'age':[], 'air_pollution':[], 'alcohol_use':[], 'dust_allergy':[], 'occ_haz':[], 'genetic_risk':[], 'ch_ld':[], 'fatigue':[], 'weight_loss':[], 'sh_breath':[], 'wheezing':[], 'swall_dif':[], 'cl_fn':[], 'freq_cold':[], 'dry_cough':[], 'snoring':[], 'balanced_diet':[], 'obesty':[], 'smoking':[], 'passive_smoker':[], 'chest_pain':[], 'coughing_of_blood':[]}        
        data['age'].append(int(request.POST.getlist('age')[0]))
        data['air_pollution'].append(int(request.POST.get('airPollution')[0]))
        data['alcohol_use'].append(int(request.POST.get('alcoholUse')[0]))
        data['dust_allergy'].append(int(request.POST.get('dustAllergy')[0]))
        data['occ_haz'].append(int(request.POST.get('occupationalHazards')[0]))
        data['genetic_risk'].append(int(request.POST.get('geneticRisk')[0]))
        data['ch_ld'].append(int(request.POST.get('chronicLungDisease')[0]))
        data['fatigue'].append(int(request.POST.get('fatigue')[0]))
        data['weight_loss'].append(int(request.POST.get('weightLoss')[0]))
        data['sh_breath'].append(int(request.POST.get('shortnessOfBreath')[0]))
        data['wheezing'].append(int(request.POST.get('wheezing')[0]))
        data['swall_dif'].append(int(request.POST.get('swallowingDifficulty')[0]))
        data['cl_fn'].append(int(request.POST.get('clubbingOfFingerNails')[0]))
        data['freq_cold'].append(int(request.POST.get('frequentCold')[0]))
        data['dry_cough'].append(int(request.POST.get('dryCough')[0]))
        data['snoring'].append(int(request.POST.get('snoring')[0]))
        data['balanced_diet'].append(int(request.POST.get('balanced_diet')[0]))
        data['obesty'].append(int(request.POST.get('obesty')[0]))
        data['smoking'].append(int(request.POST.get('smoking')[0]))
        data['passive_smoker'].append(int(request.POST.get('passive_smoker')[0]))
        data['chest_pain'].append(int(request.POST.get('chest_pain')[0]))
        data['coughing_of_blood'].append(int(request.POST.get('coughing_of_blood')[0]))
        df = pd.DataFrame(data)
        df.columns = list(features.columns)
        pred_lr = lr.predict(df)
        messages.info(request, 'Prediction is '+pred_lr[0])
    return render(request, 'comp/can.html')