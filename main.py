import numpy as np
import pandas as pd
import pickle

# Cleaning data

data = pd.read_csv('laptop_data.csv')
data = data.drop(['TypeName', 'Unnamed: 0'], axis=1)
data['Ram'] = data['Ram'].str.replace('GB', '')
data['Ram'] = data['Ram'].astype('int32')
data['Weight'] = data['Weight'].str.replace('kg', '')
data['Weight'] = data['Weight'].astype('float32')

data['TouchScreen'] = data['ScreenResolution'].apply(lambda element: 1
if 'Touchscreen' in element else 0)

data['IPS'] = data['ScreenResolution'].apply(lambda element: 1
if 'IPS Panel' in element else 0)

split = data['ScreenResolution'].str.split('x', n=1, expand=True)

data['x_res'] = split[0]
data['y_res'] = split[1]
data['x_res'] = data['x_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])

data['x_res'] = data['x_res'].astype('int32')
data['y_res'] = data['y_res'].astype('int32')

data['PPI'] = ((data['x_res'] ** 2 + data['y_res'] ** 2) ** 0.5 / data['Inches']).astype('float32')

data = data.drop(['ScreenResolution', 'x_res', 'y_res', 'Inches'], axis=1)
data['Cpu_name'] = data['Cpu'].apply(lambda text: ' '.join(text.split()[:3]))


def processor(text):
    if text == 'Intel Core i5' or text == 'Intel Core i7' or text == 'Intel Core i3':
        return text
    elif text.split()[0] == 'Intel':
        return 'other'
    else:
        return 'AMD'


data['Cpu_name'] = data['Cpu_name'].apply(lambda text: processor(text))
data = data.drop(['Cpu'], axis=1)

data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)
data['Memory'] = data['Memory'].str.replace('GB', ' ')
data['Memory'] = data['Memory'].str.replace('TB', '000')
newdf = data['Memory'].str.split('+', n=1, expand=True)

data['first'] = newdf[0]
data['first'] = data['first'].str.strip()


def change(value):
    data['layer1' + value] = data['first'].apply(lambda x: 1 if value in x else 0)


list = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']
for value in list:
    change(value)

data['first'] = data['first'].str.replace(r'\D', '')

data['second'] = newdf[1]


def change1(value):
    data['layer2' + value] = data['second'].apply(lambda x: 1 if value in x else 0)


list = ['SSD', 'HDD', 'Flash Storage', 'Hybrid']
data['second'] = data['second'].fillna('0')
for value in list:
    change1(value)

data['second'] = data['second'].str.replace(r'\D', '')
data['second'] = data['second'].astype('int32')
data['first'] = data['first'].astype('int32')

data['HDD'] = (data['first'] * data['layer1HDD'] + data['second'] * data['layer2HDD'])
data['SSD'] = (data['first'] * data['layer1SSD'] + data['second'] * data['layer2SSD'])
data['Flash Storage'] = (data['first'] * data['layer1Flash Storage'] + data['second'] * data['layer2Flash Storage'])
data['Hybrid'] = (data['first'] * data['layer1Hybrid'] + data['second'] * data['layer2Hybrid'])

data = data.drop(
    ['Flash Storage', 'Hybrid', 'first', 'second', 'layer1HDD', 'layer1SSD', 'layer1Hybrid', 'layer1Flash Storage',
     'layer2HDD', 'layer2Flash Storage', 'layer2Hybrid', 'layer2SSD'], axis=1)
data['Gpu_brand'] = data['Gpu'].apply(lambda x: x.split()[0])

data = data.drop(['Gpu'], axis=1)
data['Gpu_brand'].value_counts()


def os(text):
    if text == 'Windows 10' or text == 'Windows 7' or text == 'Windows 10s':
        return 'Windows'
    elif text == 'Mac OS X' or text == 'macOS':
        return 'mac'
    else:
        return 'other'


data['Op'] = data['OpSys'].apply(lambda x: os(x))

# Create x and y
data = data.drop(['Memory', 'OpSys'], axis=1)
x = data.drop(['Price'], axis=1)
y = data.iloc[:, 3].values

# one_hot_encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

step_1 = ColumnTransformer(transformers=[('onehotencoder', OneHotEncoder(sparse=False, drop='first'), [0, 6, 9, 10])],
                           remainder='passthrough')
step_2 = RandomForestRegressor(n_estimators=100,
                               random_state=3,
                               max_samples=0.5,
                               max_features=0.75,
                               max_depth=15)

pipe = Pipeline([('step_1', step_1), ('step_2', step_2)])
pipe.fit(x_train, y_train)

filename = 'Lap_pred.pkl'
pickle.dump(pipe, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


Company = ""



@app.route("/", methods=['POST'])
def bFlipper():
        Input = []
        Company = request.form.get('company_name')
        Ram = request.form.get('Ram')
        weight = request.form.get('Weight')
        Touchscreen = request.form.get('Touchscreen')
        if Touchscreen == 'Yes':
            Touchscreen = '1'
        else:
            Touchscreen = '0'
        IPS = request.form.get('IPS')
        if IPS == 'Yes':
            IPS = '1'
        else:
            IPS = '0'
        HDD = request.form.get('HDD')
        SSD = request.form.get('SSD')
        PPI = request.form.get('PPI')
        CPU = request.form.get('CPU')
        GPU = request.form.get('GPU')
        OS = request.form.get('OS')
        Input.append(Company)
        Input.append(Ram)
        Input.append(weight)
        Input.append(Touchscreen)
        Input.append(IPS)
        Input.append(PPI)
        Input.append(CPU)
        Input.append(HDD)
        Input.append(SSD)
        Input.append(GPU)
        Input.append(OS)

        y_prediction = loaded_model.predict([Input])
        print(y_prediction)
        return 'The Price Prediction is: {}'.format(y_prediction)


if __name__ == "__main__":
    app.run()
