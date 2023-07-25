import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/model.pkl','rb'))
standard_scaler=pickle.load(open('models/preprocessor.pkl','rb'))
# pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/pop')
def index():
    return '<h1>hiiiiiiiiiiii</h1>'

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        CIC0 = float(request.form.get('CIC0'))
        SM1_Dz = float(request.form.get('SM1_Dz'))
        GATS1i = float(request.form.get('GATS1i'))
        NdsCH = float(request.form.get('NdsCH'))
        NdssC = float(request.form.get('NdssC'))
        MLOGP = float(request.form.get('MLOGP'))
        
        
        
#         x = float(request.form.get('x'))
#         y= float(request.form.get('y'))
#         z = float(request.form.get('z'))
       

#         # stroke = float(request.form.get('stroke'))

#         # compressionratio = float(request.form.get('compressionratio'))
#         # horsepower = float(request.form.get('horsepower'))
#         # peakrpm = float(request.form.get('peakrpm'))
#         # citympg = float(request.form.get('citympg'))
#         # highwaympg = float(request.form.get('highwaympg'))
#         # fueltype_diesel = float(request.form.get('fueltype_diesel'))
#         # fueltype_gas = float(request.form.get('fueltype_gas'))
#         # aspiration_std = float(request.form.get('aspiration_std'))

#         # aspiration_turbo = float(request.form.get('aspiration_turbo'))
#         # doornumber_four = float(request.form.get('doornumber_four'))
#         # doornumber_two = float(request.form.get('doornumber_two'))
#         # drivewheel_4wd = float(request.form.get('drivewheel_4wd'))
#         # drivewheel_fwd = float(request.form.get('drivewheel_fwd'))
#         # drivewheel_rwd = float(request.form.get('drivewheel_rwd'))
#         # enginelocation_front = float(request.form.get('enginelocation_front'))
#         # enginelocation_rear = float(request.form.get('enginelocation_rear'))


        q=pd.DataFrame({'CIC0':[CIC0] ,'SM1_Dz':[SM1_Dz],'GATS1i':[GATS1i],'NdsCH':[NdsCH],'NdssC':[NdssC],'MLOGP':[MLOGP]})
        new_data_scaled=standard_scaler.transform(q)
        result=model.predict(new_data_scaled)
        # if(result[0]==0):
        #     result='P'
        # else:
        #     result='E'
        return render_template('LC50 VALUE PREDICTION.html',result=np.round(result,2))
    else:  
        return render_template('LC50 VALUE PREDICTION.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
