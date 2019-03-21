from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np
from flask import redirect, url_for

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_power', methods=['POST', 'GET'])
def predict_power():
    # get the parameters
    day_1 = float(request.form['shift_1'])
    day_2 = float(request.form['shift_2'])
    day_3 = float(request.form['shift_3'])
    day_4 = float(request.form['shift_4'])
    day_5 = float(request.form['shift_5'])
    day_6 = float(request.form['shift_6'])
    day_7 = float(request.form['shift_7'])

    # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    
    df_prediction.at[0, 'shift_1'] = day_1
    df_prediction.at[0, 'shift_2'] = day_2
    df_prediction.at[0, 'shift_3'] = day_3
    df_prediction.at[0, 'shift_4'] = day_4
    df_prediction.at[0, 'shift_5'] = day_5
    df_prediction.at[0, 'shift_6'] = day_6
    df_prediction.at[0, 'shift_7'] = day_7
    #print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_power = prediction.round(1)[0]

    # if pet_type == 1:
    #   pet_type = 'dog'
    # elif pet_type == 2:
    #   pet_type = 'cat'

    # if pet_size == 1:
    #   pet_size = 'small'
    # elif pet_size == 2:
    #   pet_size = 'medium'
    # elif pet_size == 3:
    #   pet_size = 'large'
    # elif pet_size == 4:
    #   pet_size = 'extra large'
    # elif pet_size == 0:
    #   pet_size = 'not specified'

    # if predicted_time == 0:
    #   predicted_time = 'the same day'
    # elif predicted_time == 1:
    #   predicted_time = '1-7 days'
    # elif predicted_time == 2:
    #   predicted_time = '8-30 days'
    # elif predicted_time == 3:
    #   predicted_time = '31-90 days'
    # elif predicted_time == 4:
    #   predicted_time = 'more than 100 days'

    return render_template('results.html',
                           predicted_power= int(predicted_power)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
