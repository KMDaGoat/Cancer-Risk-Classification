#this daatset will include the symptons and environmental effects which are involved in lung cancer

import pandas as pd
import numpy as np
def seconddataset():
    dataset = pd.read_csv(r"C:\Users\aliff\Downloads\lungcancerdataset\cancer patient data sets.csv")
    xval = dataset.drop(["index" , "Patient Id" , "Gender" , "Alcohol use" , "Dust Allergy" , "Clubbing of Finger Nails" , "Frequent Cold" , "Dry Cough" , "Snoring" , "Level"] , axis = "columns")
    yval = dataset["Level"].map({'Low': 0, 'Medium': 1, 'High': 1})

    xval = xval.iloc[ : , : ]
    yval = yval.to_frame(name = "Level")
    yval.iloc[: , 0]
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values= np.nan , strategy="median")
    xval = imputer.fit_transform(xval)
    yval =imputer.fit_transform(yval)
    
    from sklearn.model_selection import train_test_split
    xtrain , xtest , ytrain , ytest = train_test_split(xval , yval , test_size = 0.2 , random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    stdscale = StandardScaler()
    xtrain = stdscale.fit_transform(xtrain)
    xtest = stdscale.fit_transform(xtest)
    
    from sklearn.linear_model import LogisticRegression
    lgr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lgr.fit(xtrain  , ytrain)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(ytest, lgr.predict(xtest))
    print(f"second dataset: {accuracy}")
    
    zvalues = []
    for zvalue in lgr.decision_function(xtest):
        zvalues.append(zvalue)

    return zvalues




