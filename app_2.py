from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


# ==========================================
# MODEL VA SCALERNI YUKLASH
# ==========================================

eng_yaxshi_model = joblib.load('sugurta_modeli_2.pkl')
scaler = joblib.load('scaler_2.pkl')


# ==========================================
# MODEL USTUNLARI
# ==========================================

ustunlar_tartibi = [
    'car_age',
    'mileage',
    'engine_size',
    'owners',
    'is_automatic',
    'has_accident',
    'fuel_type_electric',
    'fuel_type_hybrid',
    'fuel_type_petrol',
    'city_tier_tier2',
    'city_tier_tier3',
    'mileage_category_Medium',
    'mileage_category_High',
    'mileage_category_Very high',
    'price_bin'
]


# ==========================================
# MASHINA NARXINI BASHORAT QILISH
# ==========================================

def mashina_narxini_bashorat_qil(
    car_age,
    mileage,
    engine_size,
    owners,
    is_automatic,
    has_accident,
    fuel_type,
    city_tier
):

    # Mileage category
    if mileage < 20000:
        mileage_category = 'Low'

    elif mileage < 60000:
        mileage_category = 'Medium'

    elif mileage < 100000:
        mileage_category = 'High'

    else:
        mileage_category = 'Very high'


    # Yangi mashina ma'lumotlari
    yangi_mashina = pd.DataFrame([{

        'car_age': car_age,
        'mileage': mileage,
        'engine_size': engine_size,
        'owners': owners,

        'is_automatic': int(is_automatic),
        'has_accident': int(has_accident),

        'fuel_type_electric':
            1 if fuel_type == 'electric' else 0,

        'fuel_type_hybrid':
            1 if fuel_type == 'hybrid' else 0,

        'fuel_type_petrol':
            1 if fuel_type == 'petrol' else 0,

        'city_tier_tier2':
            1 if city_tier == 'tier2' else 0,

        'city_tier_tier3':
            1 if city_tier == 'tier3' else 0,

        'mileage_category_Medium':
            1 if mileage_category == 'Medium' else 0,

        'mileage_category_High':
            1 if mileage_category == 'High' else 0,

        'mileage_category_Very high':
            1 if mileage_category == 'Very high' else 0

    }])


    # Modeldagi ustunlarga moslashtirish
    yangi_mashina = yangi_mashina.reindex(
        columns=ustunlar_tartibi,
        fill_value=0
    )


    # Bashorat
    natija = eng_yaxshi_model.predict(
        yangi_mashina
    )

    return natija[0]


# ==========================================
# ASOSIY SAHIFA
# ==========================================

@app.route('/')
def hello():

    return render_template(
        'index_2.html'
    )


# ==========================================
# PREDICT
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    car_age = request.form.get(
        'car_age',
        type=float
    )

    mileage = request.form.get(
        'mileage',
        type=float
    )

    engine_size = request.form.get(
        'engine_size',
        type=float
    )

    owners = request.form.get(
        'owners',
        type=int
    )

    is_automatic = (
        request.form.get('is_automatic') == '1'
    )

    has_accident = (
        request.form.get('has_accident') == '1'
    )

    fuel_type = request.form.get(
        'fuel_type'
    )

    city_tier = request.form.get(
        'city_tier'
    )


    # Bashorat
    natija = mashina_narxini_bashorat_qil(

        car_age=car_age,
        mileage=mileage,
        engine_size=engine_size,
        owners=owners,

        is_automatic=is_automatic,
        has_accident=has_accident,

        fuel_type=fuel_type,
        city_tier=city_tier
    )


    natija = round(
        float(natija),
        2
    )


    return render_template(
        'index_2.html',
        natija=natija
    )


# ==========================================
# DASTURNI ISHGA TUSHIRISH
# ==========================================

if __name__ == '__main__':

    app.run(
        debug=True
    )