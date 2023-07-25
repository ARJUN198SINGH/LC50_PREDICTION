import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 CIC0:float,
                 SM1_Dz:float,
                 GATS1i:float,
                 NdsCH:float,
                 NdssC:float,
                 MLOGP:float
                 ):
        
        self.CIC0=CIC0
        self.SM1_Dz=SM1_Dz
        self.GATS1i=GATS1i
        self.NdsCH=NdsCH
        self.NdssC=NdssC
        self.MLOGP=MLOGP
     

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CIC0':[self.CIC0],
                'SM1_Dz':[self.SM1_Dz],
                'GATS1i':[self.GATS1i],
                'NdsCH':[self.NdsCH],
                'NdssC':[self.NdssC],
                'MLOGP':[self.MLOGP],
              
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
