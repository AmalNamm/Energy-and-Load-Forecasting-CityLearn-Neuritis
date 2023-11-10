import os
import glob
import numpy as np 
import pandas as pd 
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
from difflib import SequenceMatcher
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss

from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
import re
from sklearn.preprocessing import StandardScaler
from my_models.base_predictor_model import BasePredictorModel

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        # Build the custom layer here if necessary
        super(BahdanauAttention, self).build(input_shape)

    def call(self, query, values):
        # Expand dimensions for broadcasting
        query_with_time_axis = tf.expand_dims(query, 1)

        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(BahdanauAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ExamplePredictorLGB(BasePredictorModel):
    
    
    def shape(lst):
        length = len(lst)
        shp = tuple(shape(sub) if isinstance(sub, list) else 0 for sub in lst)
        if any(x != 0 for x in shp):
            return length, shp
        else:
            return length
        
        
    def __init__(self, env_data, tau):

        # Check local evaluation
        self.num_buildings = env_data['num_buildings']
        self.building_names = env_data['building_names']
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.tau = tau
        
        self.feature_counter = 0
        self.steps = 48
        
        
        self.memory_dhw = [[] for _ in range(self.num_buildings)]
        self.memory_eep = [[] for _ in range(self.num_buildings)]
        self.memory_cl  = [[] for _ in range(self.num_buildings)]
        self.memory_sg  = []
        self.memory_cip = []

        
        # Load in pre-computed prediction model.
        self.load()
        # ====================================================================
        # insert your loading code here
        # ====================================================================

        # Create buffer/tracking attributes
        self.prev_observations = None
        self.buffer = {'key': []}
        # ====================================================================
        # dummy forecaster buffer - delete for your implementation
        # ====================================================================
        self.prev_vals = {
            **{b_name: {
                'Equipment_Eletric_Power': None,
                'DHW_Heating': None,
                'Cooling_Load': None
                } for b_name in self.building_names},
            'Solar_Generation': None,
            'Carbon_Intensity': None
        }
        self.b0_pv_capacity = env_data['b0_pv_capacity']
        # ====================================================================
        
    # Here I have to load the Prediction Model!
    def load(self):

        # LGB Models (with lags!)
        self.model_dhw_b1_GBM = joblib.load('my_models/models/LightGBM/dhw_demand_model_b1_hyper.h5')
        self.model_dhw_b2_GBM = joblib.load('my_models/models/LightGBM/dhw_demand_model_b2_hyper.h5')
        self.model_dhw_b3_GBM = joblib.load('my_models/models/LightGBM/dhw_demand_model_b3_hyper.h5')
        
        self.dhw_model_list_GBM = [self.model_dhw_b1_GBM,self.model_dhw_b2_GBM,self.model_dhw_b3_GBM]
        
        self.model_eep_b1_GBM = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b1_hyper.h5')
        self.model_eep_b2_GBM = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b2_hyper.h5')
        self.model_eep_b3_GBM = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b3_hyper.h5')
        
        self.eep_model_list_GBM = [self.model_eep_b1_GBM,self.model_eep_b2_GBM,self.model_eep_b3_GBM]
        
        
        self.model_cl_b1_GBM  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b1_hyper.h5')
        self.model_cl_b2_GBM  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b2_hyper.h5')
        self.model_cl_b3_GBM  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b3_hyper.h5')
        
        self.cl_model_list_GBM = [self.model_cl_b1_GBM,self.model_cl_b2_GBM,self.model_cl_b3_GBM]
        
        self.model_cip_GBM    = joblib.load('my_models/models/LightGBM/Carbon_Intensity_Power_model.h5')
        self.model_sg_GBM     = joblib.load('my_models/models/LightGBM/solar_generation_model.h5')
        
        
        # LGB Models (wo lags!)
        self.model_dhw_b1_GBM_wo = joblib.load('my_models/models/LightGBM/dhw_demand_model_b1_wo.h5')
        self.model_dhw_b2_GBM_wo = joblib.load('my_models/models/LightGBM/dhw_demand_model_b2_wo.h5')
        self.model_dhw_b3_GBM_wo = joblib.load('my_models/models/LightGBM/dhw_demand_model_b3_wo.h5')
        
        self.dhw_model_list_GBM_wo = [self.model_dhw_b1_GBM_wo,self.model_dhw_b2_GBM_wo,self.model_dhw_b3_GBM_wo]
        
        self.model_eep_b1_GBM_wo = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b1_wo.h5')
        self.model_eep_b2_GBM_wo = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b2_wo.h5')
        self.model_eep_b3_GBM_wo = joblib.load('my_models/models/LightGBM/Equipment_Electric_Power_model_b3_wo.h5')
        
        self.eep_model_list_GBM_wo = [self.model_eep_b1_GBM_wo,self.model_eep_b2_GBM_wo,self.model_eep_b3_GBM_wo]
        
        
        self.model_cl_b1_GBM_wo  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b1_wo.h5')
        self.model_cl_b2_GBM_wo  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b2_wo.h5')
        self.model_cl_b3_GBM_wo  = joblib.load('my_models/models/LightGBM/cooling_demand_model_b3_wo.h5')
        
        self.cl_model_list_GBM_wo = [self.model_cl_b1_GBM_wo,self.model_cl_b2_GBM_wo,self.model_cl_b3_GBM_wo]
        
        self.model_cip_GBM_wo    = joblib.load('my_models/models/LightGBM/Carbon_Intensity_Power_model_wo.h5')
        self.model_sg_GBM_wo     = joblib.load('my_models/models/LightGBM/solar_generation_model_wo.h5')
        
        
    # Make the Forecast
    def compute_forecast(self, observations):

        
        # Save the values, which are at the beginning in lists and static for each building
        feature_names_global  = ['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',  
                                  'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h', 'diffuse_solar_irradiance', 
                                  'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h', 
                                  'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h', 
                                  'direct_solar_irradiance_predicted_24h', 'carbon_intensity','electricity_pricing', 'electricity_pricing_predicted_6h', 
                                  'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h','occupant_count','net_electricity_consumption']        
       
        #----------------------------------------------------------------------------------------------------------
        # Lists to save the results of the ML models
        dhw_p_GBM = []
        eep_p_GBM = []
        cl_p_GBM  = []
        sg_p_GBM  = []
        cip_p_GBM = []
                       
        feature_values_global                                = []
        indoor_dry_bulb_temperature                          = []
        non_shiftable_load                                   = []
        solar_generation                                     = []
        dhw_storage_soc                                      = []
        electrical_storage_soc                               = []
        cooling_demand                                       = []
        dhw_demand                                           = []
        indoor_dry_bulb_temperature_set_point                = []
        
        # Store all the DFs for each Building
        building_dataframe  = []
        
        #----------------------------------------------------------------------------------------------------------
        # Iterate over the Buildings in the System and store all the necesarry information
        for i,b_name in enumerate(self.building_names):            
            
            tmp_observation_names = []
            for obssss in self.observation_names:
                for f_names in obssss: 
                    tmp_observation_names.append(f_names)

            for f_name in list(tmp_observation_names):
                        if f_name in feature_names_global:
                            
                            # Filling up the global feature set
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            feature_values_global.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        # Filling up the local feature sets
                        if f_name == 'indoor_dry_bulb_temperature': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            indoor_dry_bulb_temperature.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'non_shiftable_load': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            non_shiftable_load.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'solar_generation': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            solar_generation.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'dhw_storage_soc': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            dhw_storage_soc.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'electrical_storage_soc': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            electrical_storage_soc.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'cooling_demand': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            cooling_demand.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'dhw_demand': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            dhw_demand.append(value)
                            tmp_observation_names.remove(f_name)
                            
                        if f_name == 'indoor_dry_bulb_temperature_set_point': 
                            value = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == f_name)[0][0]])
                            indoor_dry_bulb_temperature_set_point.append(value)
                            tmp_observation_names.remove(f_name)
                            
                  
            
            
            # Generating the Dataframe per Building
            b_dataframe = pd.DataFrame(columns=['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h', 
                                      'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h', 'diffuse_solar_irradiance',
                                      'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 
                                      'diffuse_solar_irradiance_predicted_24h', 'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
                                      'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h', 'carbon_intensity', 
                                      'indoor_dry_bulb_temperature', 'non_shiftable_load', 'solar_generation', 'dhw_storage_soc', 'electrical_storage_soc', 
                                      'electricity_pricing', 'electricity_pricing_predicted_6h', 
                                      'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h', 'cooling_demand',
                                      'dhw_demand','indoor_dry_bulb_temperature_set_point','occupant_count','net_electricity_consumption'])
            
           
            
            # Fill the Dataframe
            b_dataframe.at[0,'day_type']                                   = feature_values_global[0]
            b_dataframe.at[0,'hour']                                       = feature_values_global[1]
            b_dataframe.at[0,'outdoor_dry_bulb_temperature']               = feature_values_global[2]
            b_dataframe.at[0,'outdoor_dry_bulb_temperature_predicted_6h']  = feature_values_global[3]
            b_dataframe.at[0,'outdoor_dry_bulb_temperature_predicted_12h'] = feature_values_global[4]
            b_dataframe.at[0,'outdoor_dry_bulb_temperature_predicted_24h'] = feature_values_global[5]
            b_dataframe.at[0,'diffuse_solar_irradiance']                   = feature_values_global[6]
            b_dataframe.at[0,'diffuse_solar_irradiance_predicted_6h']      = feature_values_global[7]
            b_dataframe.at[0,'diffuse_solar_irradiance_predicted_12h']     = feature_values_global[8]
            b_dataframe.at[0,'diffuse_solar_irradiance_predicted_24h']     = feature_values_global[9]
            b_dataframe.at[0,'direct_solar_irradiance']                    = feature_values_global[10]
            b_dataframe.at[0,'direct_solar_irradiance_predicted_6h']       = feature_values_global[11]
            b_dataframe.at[0,'direct_solar_irradiance_predicted_12h']      = feature_values_global[12]
            b_dataframe.at[0,'direct_solar_irradiance_predicted_24h']      = feature_values_global[13]
            b_dataframe.at[0,'carbon_intensity']                           = feature_values_global[14]
            b_dataframe.at[0,'indoor_dry_bulb_temperature']                = indoor_dry_bulb_temperature[i]
            b_dataframe.at[0,'non_shiftable_load']                         = non_shiftable_load[i]
            b_dataframe.at[0,'solar_generation']                           = solar_generation[i]
            b_dataframe.at[0,'dhw_storage_soc']                            = dhw_storage_soc[i]
            b_dataframe.at[0,'electrical_storage_soc']                     = electrical_storage_soc[i]
            b_dataframe.at[0,'electricity_pricing']                        = feature_values_global[15]
            b_dataframe.at[0,'electricity_pricing_predicted_6h']           = feature_values_global[16]
            b_dataframe.at[0,'electricity_pricing_predicted_12h']          = feature_values_global[17]
            b_dataframe.at[0,'electricity_pricing_predicted_24h']          = feature_values_global[18]
            b_dataframe.at[0,'cooling_demand']                             = cooling_demand[i]
            b_dataframe.at[0,'dhw_demand']                                 = dhw_demand[i]
            b_dataframe.at[0,'indoor_dry_bulb_temperature_set_point']      = indoor_dry_bulb_temperature_set_point[i]
            b_dataframe.at[0,'electricity_pricing_predicted_24h']          = feature_values_global[18]
            b_dataframe.at[0,'occupant_count']                             = feature_values_global[19]
            b_dataframe.at[0,'net_electricity_consumption']                = feature_values_global[20]
            
            
            # Change the df typing
            b_dataframe = b_dataframe.astype(float)
            
            #----------------------------------------------------------------------------------------------------------
            # Make the scaler for the models w/o lags
            
            if self.feature_counter < self.steps:
                scaler = StandardScaler()
                latest_features_scaled = scaler.fit_transform(b_dataframe.values.reshape(1, -1))
            
            # Local Lists for the predictions
            l_dhw = []
            l_eep = []
            l_cl  = []
            
            #----------------------------------------------------------------------------------------------------------
            # Forecaster with Lags!
            
            if self.feature_counter >= self.steps:
                
                #----------------------------------------------------------------------------------------------------------
                # Forecaster with Lags!
                # DHW
                dhw_local = b_dataframe
                for j in range(1, self.steps + 1):
                    dhw_local[f'lag_{j}'] = self.memory_dhw[i][j-1]
                    
                scaler = StandardScaler()
                latest_features_scaled_dhw = scaler.fit_transform(dhw_local.values.reshape(1, -1))
                
                # EEP
                eep_local = b_dataframe
                for j in range(1, self.steps + 1):
                    eep_local[f'lag_{j}'] = self.memory_eep[i][j-1]
                    
                scaler = StandardScaler()
                latest_features_scaled_eep = scaler.fit_transform(eep_local.values.reshape(1, -1))
                
                # CL
                cl_local = b_dataframe
                for j in range(1, self.steps + 1):
                    cl_local[f'lag_{j}'] = self.memory_cl[i][j-1]
                    
                scaler = StandardScaler()
                latest_features_scaled_cl = scaler.fit_transform(cl_local.values.reshape(1, -1))
                
                # DHW
                for _ in range(self.steps):
                    if i > 2: 
                        dhw = self.dhw_model_list_GBM[i-i].predict(latest_features_scaled_dhw)[0]
                    else:
                        dhw = self.dhw_model_list_GBM[i].predict(latest_features_scaled_dhw)[0]
                        
                    latest_features_scaled_dhw = np.roll(latest_features_scaled_dhw, -1)
                    latest_features_scaled_dhw[0, -1] = dhw
                    l_dhw.append(dhw)

                # EEP
                for _ in range(self.steps):
                    if i > 2: 
                        eep = self.eep_model_list_GBM[i-i].predict(latest_features_scaled_eep)[0]
                    else:
                        eep = self.eep_model_list_GBM[i].predict(latest_features_scaled_eep)[0]
                        
                    latest_features_scaled_eep = np.roll(latest_features_scaled_eep, -1)
                    latest_features_scaled_eep[0, -1] = eep
                    l_eep.append(eep)


                # CL
                for _ in range(self.steps):
                    if i > 2: 
                        cl = self.cl_model_list_GBM[i-i].predict(latest_features_scaled_cl)[0]
                    else:
                        cl = self.cl_model_list_GBM[i].predict(latest_features_scaled_cl)[0]
                    latest_features_scaled_cl = np.roll(latest_features_scaled_cl, -1)
                    latest_features_scaled_cl[0, -1] = cl
                    l_cl.append(cl)
                    
            
            #----------------------------------------------------------------------------------------------------------
            # Forecasting w/o lags since we have to build up the memory
            
            else:
                
                # DHW
                for _ in range(self.steps):
                    if i > 2: 
                        dhw = self.dhw_model_list_GBM_wo[i-i].predict(latest_features_scaled)
                    else:
                        dhw = self.dhw_model_list_GBM_wo[i].predict(latest_features_scaled)
                    l_dhw.append(dhw)

                # EEP
                for _ in range(self.steps):
                    if i > 2: 
                        eep = self.eep_model_list_GBM_wo[i-i].predict(latest_features_scaled)
                    else:
                        eep = self.eep_model_list_GBM_wo[i].predict(latest_features_scaled)
                    l_eep.append(eep)
                    
                # CL
                for _ in range(self.steps):
                    if i > 2: 
                        cl = self.cl_model_list_GBM_wo[i-i].predict(latest_features_scaled)
                    else:
                        cl = self.cl_model_list_GBM_wo[i].predict(latest_features_scaled)
                    l_cl.append(cl)

            #----------------------------------------------------------------------------------------------------------
            # Here we made the predictions etc for one building and we save the results / information of each building
            # in the following global lists
            
            building_dataframe.append(b_dataframe)
            dhw_p_GBM.append(l_dhw)
            eep_p_GBM.append(l_eep)
            cl_p_GBM.append(l_cl)
                
        #----------------------------------------------------------------------------------------------------------
        # Here we concat the Building Datasets to one, since for the global features sg and ci are on a neighbour
        # level prediction!
        
        comb_dataframe = pd.concat(building_dataframe, ignore_index=True)
        comb_dataframe.reset_index(drop=True, inplace=True)
        comb_dataframe = comb_dataframe.mean(axis=0)
        
        #----------------------------------------------------------------------------------------------------------
        # Filling the memory with the lagged features
        
        if self.feature_counter < self.steps:
            for idx,building in enumerate(building_dataframe):
                self.memory_dhw[idx].append(building['dhw_demand'])
                self.memory_eep[idx].append(building['non_shiftable_load'])
                self.memory_cl[idx].append(building['cooling_demand'])
            self.memory_sg.append(comb_dataframe['solar_generation'])
            self.memory_cip.append(comb_dataframe['carbon_intensity'])
        
        #----------------------------------------------------------------------------------------------------------    
        # Make the neighbour level predictions
        
        # First scale the combined Dataset of the buildings in the system
        scaler = StandardScaler()
        latest_features_scaled_combined  = scaler.fit_transform(comb_dataframe.values.reshape(1, -1))
        
        #----------------------------------------------------------------------------------------------------------
        # Neighbour level Predictions with lags
        
        if self.feature_counter >= self.steps:
            # SG
            sg_local = comb_dataframe
            
            for j in range(1, self.steps + 1):
                sg_local[f'lag_{j}'] = self.memory_sg[j-1]
            scaler = StandardScaler()
            latest_features_scaled_combined_sg = scaler.fit_transform(sg_local.values.reshape(1, -1))
            
            # CIG
            cip_local = comb_dataframe
            for j in range(1, self.steps + 1):
                cip_local[f'lag_{j}'] = self.memory_cip[j-1]
                    
            scaler = StandardScaler()
            latest_features_scaled_combined_cip = scaler.fit_transform(cip_local.values.reshape(1, -1))
                
            # SG
            for _ in range(self.steps):
                sg  = self.model_sg_GBM.predict(latest_features_scaled_combined_sg)[0]

                latest_features_scaled_combined_sg = np.roll(latest_features_scaled_combined_sg, -1)
                latest_features_scaled_combined_sg[0, -1] = sg
                sg_p_GBM.append(sg)

            # CIG
            for _ in range(self.steps):
                cip = self.model_cip_GBM.predict(latest_features_scaled_combined_cip)[0]

                latest_features_scaled_combined_cip = np.roll(latest_features_scaled_combined_cip, -1)
                latest_features_scaled_combined_cip[0, -1] = cip
                cip_p_GBM.append(cip) 
                
        #----------------------------------------------------------------------------------------------------------
        # Neighbour level Predictions w/o lags
        
        else: 
            # SG
            for _ in range(self.steps):
                sg  = self.model_sg_GBM_wo.predict(latest_features_scaled_combined)
                sg_p_GBM.append(sg)

            # CIG
            for _ in range(self.steps):
                cip = self.model_cip_GBM_wo.predict(latest_features_scaled_combined)
                cip_p_GBM.append(cip)     
            
        #----------------------------------------------------------------------------------------------------------
        # Dictionary of the current values
        
        current_vals = {
            **{b_name: {
                'Equipment_Eletric_Power': 
             np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'non_shiftable_load')[0][i]],
                
                'DHW_Heating': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'dhw_demand')[0][i]],
                
                'Cooling_Load': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'cooling_demand')[0][i]]
                } for i,b_name in enumerate(self.building_names)},
            'Solar_Generation': 
        np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'solar_generation')[0][0]]/self.b0_pv_capacity*1000,
            'Carbon_Intensity': np.array(observations)[0][np.where(np.array(self.observation_names)[0] == 'carbon_intensity')[0][0]]
        }
        
        #----------------------------------------------------------------------------------------------------------
        # Dictionary of the prediction values
        
        if self.prev_vals['Carbon_Intensity'] is None:
            predictions_dict = {
                **{b_name: {
                    'Equipment_Eletric_Power': [current_vals[b_name]['Equipment_Eletric_Power'] for _ in range(self.tau)],
                    'DHW_Heating': [current_vals[b_name]['DHW_Heating'] for _ in range(self.tau)],
                    'Cooling_Load': [current_vals[b_name]['Cooling_Load'] for _ in range(self.tau)]
                    } for i,b_name in enumerate(self.building_names)},
                'Solar_Generation': [current_vals['Solar_Generation'] for _ in range(self.tau)],
                'Carbon_Intensity': [current_vals['Carbon_Intensity'] for _ in range(self.tau)]
            }

        else:
                predictions_dict = {}
                predict_inds = [t+1 for t in range(self.tau)]

                for i,b_name in enumerate(self.building_names):
                    predictions_dict[b_name] = {}
                
                    for load_type in ['Equipment_Eletric_Power','DHW_Heating','Cooling_Load']:
                    
                        if load_type == 'Equipment_Eletric_Power':
                            predictions_dict[b_name][load_type] = eep_p_GBM[i]
                            
                        if load_type == 'DHW_Heating':
                            predictions_dict[b_name][load_type] = dhw_p_GBM[i]
                            
                        if load_type == 'Cooling_Load':
                            predictions_dict[b_name][load_type] = cl_p_GBM[i]            
                
                predictions_dict['Solar_Generation'] = sg_p_GBM
                predictions_dict['Carbon_Intensity'] = cip_p_GBM

                
        self.feature_counter = self.feature_counter + 1
        self.prev_vals = current_vals
        return predictions_dict
    
    
    
    
    
