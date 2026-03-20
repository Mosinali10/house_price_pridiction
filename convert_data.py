import pandas as pd, json
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv('data/housing.csv', sep=r'\s+', names=cols)
with open('data/housing.json', 'w') as f:
    json.dump(df.to_dict(orient='records'), f)
print("Done", len(df), "rows")
