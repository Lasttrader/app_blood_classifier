import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import random

app = Flask(__name__)
model = pickle.load(open('KNN01.pkl', 'rb'))
# Ввод даных 
analysis_list =[]
full_list =[]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # создаем список
    demo_list =[]
    analysis_list = []
    full_list =[]
    x_list = []


    demo_list.clear()# очистим список
    RIAGENDR =  request.form['gender']
    RIDAGEMN =  request.form['age']
    INDFMINC =  request.form['famlyincome']
    INDHHINC =  request.form['hincome']
    DMDMARTL =  request.form['marital']
    RIDRETH1 =  request.form['race']
    INDFMPIR =  request.form['poverty']
    # Добавляем  список демографии
    demo_list.extend([RIAGENDR, RIDAGEMN,   INDFMINC,   INDHHINC,   DMDMARTL,   RIDRETH1,   INDFMPIR])

    LBDBANO = random.uniform(5.397605e-79, 4.700000e+00)
    LBDEONO = random.uniform(5.397605e-79, 8.400000e+00)
    LBDLYMNO = random.uniform(0.2, 110.8)
    LBDMONO = random.uniform(5.397605e-79, 1.020000e+01)
    LBDNENO = random.uniform(0.1, 83.1)
    LBXBAPCT = random.uniform(5.397605e-79, 3.540000e+01)   
    LBXEOPCT = random.uniform(5.397605e-79, 5.720000e+01)
    LBXHCT = random.uniform(16.3, 59.9)
    LBXHGB = random.uniform(5.8, 19.7)  
    LBXLYPCT = random.uniform(2.6, 94.5)    
    LBXMC = random.uniform(25.1, 69.6)
    LBXMCHSI = random.uniform(13.8, 74.5)   
    LBXMCVSI = random.uniform(50.5, 125.3)  
    LBXMOPCT = random.uniform(0.6, 66.9)    
    LBXMPSI =   random.uniform(4.7, 15.1)
    LBXNEPCT = random.uniform(0.8, 96.6)    
    LBXPLTSI = random.uniform(4.0, 1000.0)  
    LBXRBCSI = random.uniform(1.67, 9.16)   
    LBXRDW =    random.uniform(6.30, 37.80)
    LBXWBCSI = random.uniform(1.40, 117.20) 
    LBDSALSI = random.uniform(12.00, 57.00) 
    LBDSBUSI = random.uniform(0.36, 43.550) 
    LBDSCASI = random.uniform(1.6250, 3.70) 
    LBDSCHSI = random.uniform(0.1550, 18.4120)  
    LBDSCRSI = random.uniform(8.8400, 1573.520) 
    LBDSGBSI = random.uniform(6.00, 79.00)  
    LBDSGLSI = random.uniform(1.050, 43.130)    
    LBDSIRSI = random.uniform(0.40, 99.80)  
    LBDSPHSI = random.uniform(0.3230, 3.520)    
    LBDSTBSI = random.uniform(5.397605e-79, 2.240100e+02)   
    LBDSTPSI = random.uniform(34.00, 113.00)    
    LBDSTRSI = random.uniform(0.1020, 68.3840)  
    LBDSUASI = random.uniform(23.80, 1070.60)   
    LBXSAL =    random.uniform(1.20, 5.70)
    LBXSAPSI = random.uniform(7.00, 1378.00)    
    LBXSASSI = random.uniform(7.00, 1672.00)    
    LBXSATSI = random.uniform(3.00, 1997.00)    
    LBXSBU =    random.uniform(1.00, 122.00)
    LBXSC3SI = random.uniform(10.00, 43.00) 
    LBXSCA =    random.uniform(6.50, 14.80)
    LBXSCH =    random.uniform(6.00, 712.00)
    LBXSCLSI = random.uniform(70.00, 120.00)    
    LBXSCR =    random.uniform(0.1400, 17.80)
    LBXSGB =    random.uniform(0.60, 7.90)
    LBXSGL =    random.uniform(19.00, 777.00)
    LBXSGTSI = random.uniform(3.00, 2274.00)
    LBXSIR = random.uniform(2.00, 557.00)   
    LBXSKSI =   random.uniform(2.30, 7.30)
    LBXSLDSI = random.uniform(4.00, 1539.00)
    LBXSNASI = random.uniform(99.00, 161.00)
    LBXSOSSI = random.uniform(201.00, 323.00)
    LBXSPH = random.uniform(1.00, 10.90)
    LBXSTB = random.uniform(5.397605e-79, 1.310000e+01)
    LBXSTP = random.uniform(3.40, 11.30)
    LBXSTR = random.uniform(9.00, 6057.00)
    LBXSUA = random.uniform(0.40, 18.00)

    # Анализы
    # Получаем отрицательные анализы
    annegative = request.form['annegative'] 
    if annegative == '1':
        # очистим список
        analysis_list.clear()
        # добавляем анализы в список
        analysis_list.extend ([LBDBANO, LBDEONO, LBDLYMNO, LBDMONO, LBDNENO,    LBXBAPCT,   LBXEOPCT,   LBXHCT, LBXHGB, LBXLYPCT,   LBXMC,  LBXMCHSI,   LBXMCVSI,   LBXMOPCT,   LBXMPSI,    LBXNEPCT,   LBXPLTSI,   LBXRBCSI,   LBXRDW, LBXWBCSI,   LBDSALSI,   LBDSBUSI,   LBDSCASI,   LBDSCHSI,   LBDSCRSI,   LBDSGBSI,   LBDSGLSI,   LBDSIRSI,   LBDSPHSI,   LBDSTBSI,   LBDSTPSI,   LBDSTRSI,   LBDSUASI,   LBXSAL, LBXSAPSI,   LBXSASSI,   LBXSATSI,   LBXSBU, LBXSC3SI,   LBXSCA, LBXSCH, LBXSCLSI,   LBXSCR, LBXSGB, LBXSGL, LBXSGTSI,   LBXSIR, LBXSKSI,    LBXSLDSI,   LBXSNASI,   LBXSOSSI,   LBXSPH, LBXSTB, LBXSTP, LBXSTR, LBXSUA])
    
    # Получаем положительные анализы
    #2.0    920.000000  5.000000    6.000000    2.0 3.0 2.400000
    anpositive = request.form['anpositive'] 
    if anpositive == '1':
        # очистим список
        analysis_list.clear()
        # добавляем анализы в список
        analysis_list.extend([5.397605e-79, 0.9, 1.1, 0.5, 5.8, 0.6, 10.4, 32.8, 
                              11.2, 13.8, 34.300000, 32.4, 94.5, 5.5, 8.1, 69.7, 
                              263.0, 3.47, 11.8, 8.3, 41.0, 13.60, 2.450, 5.070, 
                              114.90, 36.0, 4.885, 8.24, 1.001, 6.80, 77.0, 1.185,
                              624.5, 4.1, 128.0, 17.0, 9.0, 38.0, 20.0, 9.8, 196.0,   
                              103.3, 1.30, 3.6, 88.0, 14.0, 46.0, 4.78, 164.0, 138.3, 
                              285.0, 3.1, 0.4, 7.7, 105.0, 10.5])
    
    anrandom = request.form['anrandom'] 
    if anrandom == '1':
        # очистим список
        analysis_list.clear()
        # добавляем анализы в список
        analysis_list.extend([LBDBANO, LBDEONO, LBDLYMNO, LBDMONO, LBDNENO,    LBXBAPCT,   LBXEOPCT,   LBXHCT, LBXHGB, LBXLYPCT,   LBXMC,  LBXMCHSI,   LBXMCVSI,   LBXMOPCT,   LBXMPSI,    LBXNEPCT,   LBXPLTSI,   LBXRBCSI,   LBXRDW, LBXWBCSI,   LBDSALSI,   LBDSBUSI,   LBDSCASI,   LBDSCHSI,   LBDSCRSI,   LBDSGBSI,   LBDSGLSI,   LBDSIRSI,   LBDSPHSI,   LBDSTBSI,   LBDSTPSI,   LBDSTRSI,   LBDSUASI,   LBXSAL, LBXSAPSI,   LBXSASSI,   LBXSATSI,   LBXSBU, LBXSC3SI,   LBXSCA, LBXSCH, LBXSCLSI,   LBXSCR, LBXSGB, LBXSGL, LBXSGTSI,   LBXSIR, LBXSKSI,    LBXSLDSI,   LBXSNASI,   LBXSOSSI,   LBXSPH, LBXSTB, LBXSTP, LBXSTR, LBXSUA])

    full_list.clear()
    full_list.extend(demo_list + analysis_list)

    full_list = np.array(full_list).reshape(-1, 1)
    scaler = StandardScaler()
    x = scaler.fit_transform(full_list)
    x = x.reshape(1,-1)
    x_list.extend(x)


    prediction = model.predict(x_list)

    

    return render_template('index.html', test_text2 = 'Массив для анализа: {}'.format(x_list), prediction_text=' Класс Blood classifier: {}'.format(prediction))
    

if __name__ == "__main__":
    app.run(debug=True)