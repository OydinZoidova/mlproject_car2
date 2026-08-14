
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

eng_yaxshi_model = joblib.load("sugurta_modeli.pkl")
scaler = joblib.load("scaler.pkl")

ustunlar_tartibi = [
    "age",
    "bmi",
    "children",
    "is_female",
    "is_smoker",
    "region_northwest",
    "region_southeast",
    "region_southwest",
    "bmi_category_Normal",
    "bmi_category_overweight",
    "bmi_category_obese"
]

def sugurta_narxini_bashorat_qil(
    yosh,
    bmi,
    farzandlar_soni,
    ayolmi,
    chekadimi,
    hudud,
    model,
    scaler,
    ustunlar
):
    if bmi < 18.5:
        bmi_kategoriya = "Underweight"
    elif bmi < 25:
        bmi_kategoriya = "Normal"
    elif bmi < 30:
        bmi_kategoriya = "Overweight"
    else:
        bmi_kategoriya = "Obese"

    yangi_mijoz = pd.DataFrame([{
        "age": yosh,
        "bmi": bmi,
        "children": farzandlar_soni,
        "is_female": int(ayolmi),
        "is_smoker": int(chekadimi),
        "region_northwest": 1 if hudud == "northwest" else 0,
        "region_southeast": 1 if hudud == "southeast" else 0,
        "region_southwest": 1 if hudud == "southwest" else 0,
        "bmi_category_Normal": 1 if bmi_kategoriya == "Normal" else 0,
        "bmi_category_overweight": 1 if bmi_kategoriya == "Overweight" else 0,
        "bmi_category_obese": 1 if bmi_kategoriya == "Obese" else 0
    }])

    yangi_mijoz[["age", "bmi", "children"]] = scaler.transform(
        yangi_mijoz[["age", "bmi", "children"]]
    )

    yangi_mijoz = yangi_mijoz[ustunlar]

    natija = model.predict(yangi_mijoz)

    return natija[0]

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    natija = None
    xatolik = None

    if request.method == "POST":
        try:
            yosh = request.form.get("yosh", type=float)
            bmi = request.form.get("bmi", type=float)
            farzandlar_soni = request.form.get("children", type=int)
            ayolmi = request.form.get("gender") == "1"
            chekadimi = request.form.get("chekadimi") == "1"
            hudud = request.form.get("hudud")

            if yosh is None or bmi is None or farzandlar_soni is None:
                raise ValueError("Barcha maydonlarni to'ldiring.")

            if hudud is None:
                raise ValueError("Hududni tanlang.")

            natija = sugurta_narxini_bashorat_qil(
                yosh=yosh,
                bmi=bmi,
                farzandlar_soni=farzandlar_soni,
                ayolmi=ayolmi,
                chekadimi=chekadimi,
                hudud=hudud,
                model=eng_yaxshi_model,
                scaler=scaler,
                ustunlar=ustunlar_tartibi
            )

            natija = round(float(natija), 2)

        except Exception as e:
            xatolik = str(e)

    return render_template(
        "index.html",
        natija=natija,
        xatolik=xatolik
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
