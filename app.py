from flask import Flask, render_template, request
from model import model
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    kitchen = int(request.form['kitchen'])

    storeroom = int(request.form['storeroom'])
    bighall = int(request.form['bighall'])

    parking = int(request.form['parking'])
    garden = int(request.form['garden'])
    swimmingpool = int(request.form['swimmingpool'])

    data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'kitchen': [kitchen],
        'storeroom': [storeroom],
        'bighall': [bighall],
        'parking': [parking],
        'garden': [garden],
        'swimmingpool': [swimmingpool]
    })

    prediction = model.predict(data)

    price = prediction[0]

    # Extra feature values
    if storeroom == 1:
        price += 500000

    if bighall == 1:
        price += 1000000

    if parking == 1:
        price += 700000

    if garden == 1:
        price += 800000

    if swimmingpool == 1:
        price += 2000000

    return render_template(
        'index.html',
        prediction_text=f"Estimated House Price: ₹{price:,.0f}"
    )

if __name__ == '__main__':
    app.run(debug=True)