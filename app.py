# This is the app

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session
import pickle
import ast
import json
from flask_wtf import FlaskForm
from wtforms import SelectField
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret'

df = pd.read_csv('data/heart.csv')
ChestPain = df['ChestPainType'].unique()
restingECG = df['RestingECG'].unique()
stslope = df['ST_Slope'].unique()

class Form(FlaskForm):
    Age = SelectField('Age', choices=[(champ, champ) for champ in range(100)])
    Gender = SelectField('Gender', choices=['M','F'])
    chestPain = SelectField('Chest pain type', choices=ChestPain)
    RestingBP = SelectField('Resting blood pressure', choices=[(champ, champ) for champ in np.arange(df['RestingBP'].min(),df['RestingBP'].max(),1)])
    Cholesterol = SelectField('Serum cholesterol', choices=[(champ, champ) for champ in np.arange(df['Cholesterol'].min(),df['Cholesterol'].max(),1)])
    FastingBS = SelectField('Fasting blood sugar',choices=['0','1'])
    RestingECG = SelectField('Resting electrocardiogram results', choices= restingECG)
    MaxHR = SelectField('Maximum heart rate achieved', choices=[(champ, champ) for champ in np.arange(df['MaxHR'].min(),df['MaxHR'].max(),1)])
    ExerciseAngina = SelectField('Exercise-induced angina', choices=['Y','N'])
    Oldpeak = SelectField('Oldpeak', choices=[(round(champ, 2), champ) for champ in np.arange(-10,10,0.1)])
    ST_Slope = SelectField('The slope of the peak exercise ST segment', choices = stslope)

@app.route('/', methods=['GET', 'POST'])
def home():
    is_pred = False
    print(session)
    form = Form()
    return render_template('index.html', form=form, is_pred=is_pred)

@app.route('/predict', methods=['POST'])
def predict():
    is_pred = True

    features = df.drop(columns=['HeartDisease'])
    targets = df['HeartDisease']
    features = pd.get_dummies(features) 
    feature_names = list(features.columns) #columns that have to be input into model
    print('\n',feature_names, '\n')
    user_response = list(request.form.values())
    tag_dict = {}
    print('\n',user_response, '\n')
    # for i in range(10):
    #     user_response[i] = tag_dict[str(user_response)]

    features = [int(x) if y == 1 else float(x) if y == 2 else y+x for x, y in zip(user_response, [1, 'Sex_', 'ChestPainType_', 1, 1, 1, 
                                                                            'RestingECG_', 1, 'ExerciseAngina_', 2, 'ST_Slope_'])] # add another empty
    print('\n',features,'\n')

    feature_dict = {}

    for feat, label in zip(features, ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']):
        print(label)
        feature_dict[label] = feat

    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
    float_features = ['Oldpeak']

    input_features = []

    for feat in feature_names:
        if feat in list(feature_dict.keys()):
            input_features.append(feature_dict[feat])
        else:
            if feat in list(feature_dict.values()):
                input_features.append(1)
            else:
                input_features.append(0)
    
    print('\n',input_features, '\n') # THIS IS THE INPUT TO THE MODEL MODEL.PREDICT(INPUT_FEATURES)

    model = pickle.load(open('models/BM.pkl', 'rb'))

    input_features_np = np.array(input_features)

    print(model.predict(input_features_np.reshape(1,-1)))
    print(input_features_np.reshape(1,-1))

    return render_template('models.html', prediction_text='Your probability of dying is: {}'.format(model.predict_proba(input_features_np.reshape(1,-1))[0][1]), form=Form(), is_pred=is_pred)


    # print(model)

    # for i in range(len(x.columns)):
    #     if pd.Series.between(i, 0, 3): # put back to 4
    #         pass
    #     elif x.columns[i] in features:
    #         print('champ {} in {}'.format(x.columns[i], i))
    #         input_array[i] = 1
    #     else:
    #         input_array[i] = 0
    
    # prediction = model.predict_proba([np.array(input_array)])
    # output = round(prediction[0][0], 3)
    
    # print('Your probability of winning is: {} for the input: {}'.format('output', 'features'))
    
'''        
    input_array = [0]*len(x.columns)

    for index, index2 in zip(range(5), range(10,15)):
        input_array[index] = features[index2]

    for i in range(len(x.columns)):
        if pd.Series.between(i, 0, 4):
            pass
        elif x.columns[i] in features:
            print('champ {} in {}'.format(x.columns[i], i))
            input_array[i] = 1    
        else:
            input_array[i] = 0

    prediction = model.predict_proba([np.array(input_array)])
    output = round(prediction[0][0], 3)
'''

@app.route('/Model1', methods=['POST', 'GET'])
def Model1():
    is_pred = False
    isIndex=False
    session['my_var'] = 'M1'
    form = Form()
    return render_template('models.html', form=form, isIndex=isIndex, is_pred=is_pred)

#get rid of this part, but cuztomize html so model 2 button gets removed
@app.route('/Model2', methods=['POST', 'GET'])
def Model2():
    is_pred = False
    isIndex=True
    session['my_var'] = 'M2'
    form = Form()
    return render_template('models.html', form=form, isIndex=isIndex, is_pred=is_pred)


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)