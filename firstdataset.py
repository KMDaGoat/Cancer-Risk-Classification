import pandas as pd
import numpy as np
def firstdataset():
    dataset = pd.read_csv(r"C:\Users\aliff\Downloads\lungcancerdataset\lung_cancer_dataset.csv")
    dataset.drop(["patient_id", "secondhand_smoke_exposure", "copd_diagnosis"], axis="columns", inplace=True)
    
    yval = dataset["lung_cancer"]
    xval = dataset.drop("lung_cancer", axis="columns")

    xval["gender"] = xval["gender"].map({'Female': 0, 'Male': 1})
    xval["family_history"] = xval["family_history"].map({'No': 0, 'Yes': 1})
    
    yval = yval.map({'No': 0, 'Yes': 1})

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder , LabelEncoder
    
    aimedcategories = ["radon_exposure" , "alcohol_consumption" , "asbestos_exposure"]
    columntransform = ColumnTransformer(
        transformers= [("cat" , OneHotEncoder(handle_unknown="ignore") , aimedcategories)],remainder = "passthrough")

    xvalencoded = columntransform.fit_transform(xval)
    feature_names = columntransform.get_feature_names_out()
    xvalencoded = pd.DataFrame(xvalencoded, columns=feature_names)

    x = xvalencoded.iloc[:, :]
    y = yval.to_frame(name='lung_cancer') 
    y.iloc[:, 0] 

    # now to fill in the missing data values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x = imputer.fit_transform(x)
    y = imputer.fit_transform(y)

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

    from sklearn.preprocessing import StandardScaler
    standardscale = StandardScaler()
    xtrain = standardscale.fit_transform(xtrain)
    xtest = standardscale.fit_transform(xtest)


    from sklearn.linear_model import LogisticRegression
    lgr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lgr.fit(xtrain, ytrain)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(ytest, lgr .predict(xtest))
    print(f"first dataset: {accuracy}")

    zvalues = []
    for zvalue in lgr.decision_function(xtest):
        zvalues.append(zvalue)

    return zvalues


