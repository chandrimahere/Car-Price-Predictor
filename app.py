from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        company = request.form['company']
        year = request.form['year']
        kms = request.form['kms']
        fuel = request.form['fuel']
        year = int(year)
        kms = int(kms)
        fuel = int(fuel)

        price = predictor(company, year, kms, fuel)
        return str(price[0])
    return render_template('index.html')


def predictor(company, year, kms, fuel):
    data = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1jDfMNCRqFLTnGFRK3tq7QhcbAnQe9SHXwlzlaScMU1c/export?format=csv")
    for i in range(len(data['Price'])):
        value = 0
        if data['Price'][i] != 'Ask For Price':
            data['Price'][i] = str(data['Price'][i])
            for j in data['Price'][i]:
                if j != ',':
                    value = value * 10 + int(j)
            data['Price'][i] = value
    data['Price'].replace('Ask For Price', np.nan, inplace=True)
    data['Price'].fillna(data['Price'].mean(), inplace=True)
    data['kms_driven'].fillna('0 kms', inplace=True)
    data.loc[data.tail(2).index, 'kms_driven'] = '0 kms'
    data.loc[data.tail(2).index, 'fuel_type'] = 'Petrol'
    data['kms_driven'] = data['kms_driven'].str.replace(' kms', '')
    data['kms_driven'] = data['kms_driven'].str.replace(',', '')
    data['kms_driven'] = pd.to_numeric(data['kms_driven'])
    data['fuel_type'].fillna('Other', inplace=True)
    data['fuel_type'] = data['fuel_type'].map({'Petrol': 1, 'Diesel': 2, 'LPG': 3, 'Other': 0})
    mask = data['year'].str.startswith('20') == False
    data['year'][mask] = np.nan
    data['year'] = pd.to_numeric(data['year'])
    m = int(data['year'].mean())
    data['year'].fillna(m, inplace=True)
    data['year'] = data['year'].astype('int64')
    data['fuel_type'] = data['fuel_type'].astype('category')
    s = data['company'].value_counts()
    data['company'] = np.where(data['company'].isin(s.index[s >= 4]), data['company'], 'Other')
    data['company'] = data['company'].astype('category')
    company_index = data['company'].value_counts().index
    company_array = []
    for i in data['company']:
        t = []
        for j in company_index:
            if i == j:
                t.append(1)
            else:
                t.append(0)
        company_array.append(t)
    company_array = np.array(company_array)
    for i in range(len(company_index)):
        data[company_index[i]] = company_array[:, i]
    data.drop(columns=['name', 'company'], inplace=True)
    col = data.columns.tolist()
    col = col[1:2] + col[0:1] + col[2:]
    data = data[col]
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    test = np.array([year, kms, fuel])

    for i in range(4, len(data.columns)):
        if data.columns[i] == company:
            test = np.append(test, [1])
        else:
            test = np.append(test, [0])

    test_final = pd.DataFrame(test.reshape(-1, len(test)))
    return regressor.predict(test_final)


if __name__ == "__main__":
    app.run(debug=True)