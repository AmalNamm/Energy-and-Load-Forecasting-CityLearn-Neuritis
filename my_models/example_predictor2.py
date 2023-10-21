"""
Implementation of example prediction method.
Note, this forecasting method is illustratively *terrible*, but
provides a reference on how to format your forecasting method.

The forecasting/prediction model is implemented as a class.

This class must have the following methods:
    - __init__(self, ...), which initialises the Predictor object and
        performs any initial setup you might want to do.
    - compute_forecast(observation), which executes your prediction method,
        creating timeseries forecasts for [building electrical loads,
        normalise solar pv generation powers, grid carbon intensity]
        given the current observation.

You may wish to implement additional methods to make your model code neater.
"""

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


from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss

from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
import re

from my_models.base_predictor_model import BasePredictorModel


class ExamplePredictor2(BasePredictorModel):

    
    def shape(lst):
        length = len(lst)
        shp = tuple(shape(sub) if isinstance(sub, list) else 0 for sub in lst)
        if any(x != 0 for x in shp):
            return length, shp
        else:
            return length
        
        
    def __init__(self, env_data, tau):
        """Initialise Prediction object and perform setup.
        
        Args:
            env_data : Dictionary containing data about the environment 
                    observation_names = env.observation_names,
                    observation_space = env.observation_space,
                    action_space = env.action_space,
                    time_steps = env.time_steps,
                    buildings_metadata = env.get_metadata()['buildings'],
                    num_buildings = len(env.buildings),
                    building_names = [b.name for b in env.buildings],
                    b0_pv_capacity = env.buildings[0].pv.nominal_power,
                    
            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
        """

        # Check local evaluation
        self.num_buildings = env_data['num_buildings']
        self.building_names = env_data['building_names']
        self.observation_names = env_data['observation_names']
        self.action_names = env_data['action_names']
        self.tau = tau

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

        self.model_dhw_b1 = joblib.load('my_models/models/dhw_load_model_b1_new_2.pkl')
        self.model_dhw_b2 = joblib.load('my_models/models/dhw_load_model_b2_new_2.pkl')
        self.model_dhw_b3 = joblib.load('my_models/models/dhw_load_model_b3_new_2.pkl')
        
        self.dhw_model_list = [self.model_dhw_b1,self.model_dhw_b2,self.model_dhw_b3]
        
      
        self.model_sg_b1  = joblib.load('my_models/models/solar_generation_model_b1_new_2.pkl')
        self.model_sg_b2  = joblib.load('my_models/models/solar_generation_model_b2_new_2.pkl')
        self.model_sg_b3  = joblib.load('my_models/models/solar_generation_model_b3_new_2.pkl')
        
        self.sg_model_list = [self.model_sg_b1,self.model_sg_b2,self.model_sg_b3]
        
        self.model_eep_b1 = joblib.load('my_models/models/Equipment_Electric_Power_model_b1_new_2.pkl')
        self.model_eep_b2 = joblib.load('my_models/models/Equipment_Electric_Power_model_b2_new_2.pkl')
        self.model_eep_b3 = joblib.load('my_models/models/Equipment_Electric_Power_model_b3_new_2.pkl')
        
        self.eep_model_list = [self.model_eep_b1,self.model_eep_b2,self.model_eep_b3]
        
        self.model_cl_b1  = joblib.load('my_models/models/cooling_load_model_b1_new_2.pkl')
        self.model_cl_b2  = joblib.load('my_models/models/cooling_load_model_b1_new_2.pkl')
        self.model_cl_b3  = joblib.load('my_models/models/cooling_load_model_b1_new_2.pkl')
        
        self.cl_model_list = [self.model_cl_b1,self.model_cl_b2,self.model_cl_b3]
        
        self.model_cip    = joblib.load('my_models/models/Carbon_Intensity_Power_model_new_2.pkl')
        
    def compute_forecast(self, observations):

        
        # Features of the Predictors: 
        
        #'day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h', 
        #'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h', 'diffuse_solar_irradiance',
        #'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 
        #'diffuse_solar_irradiance_predicted_24h', 'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
        #'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h', 'carbon_intensity', 
        #'indoor_dry_bulb_temperature', 'non_shiftable_load', 'solar_generation', 'dhw_storage_soc', 'electrical_storage_soc', 
        #'electricity_pricing', 'electricity_pricing_predicted_6h', 
        #'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h', 'cooling_demand',
        #'dhw_demand','indoor_dry_bulb_temperature_set_point'
        
        # A total of 27 Available Features! (Right now all are in use!)
        
        # Dynamic Code for the Forecasting
        
        # Save the values, which are at the beginning in lists and static for each building
        feature_names_global  = ['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',  
                                  'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h', 'diffuse_solar_irradiance', 
                                  'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h', 
                                  'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h', 
                                  'direct_solar_irradiance_predicted_24h', 'carbon_intensity','electricity_pricing', 'electricity_pricing_predicted_6h', 
                                  'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h','occupant_count','net_electricity_consumption']        

        
        building_numbers = len(self.building_names)
       
        feature_values_global = []
        indoor_dry_bulb_temperature                          = []
        non_shiftable_load                                   = []
        solar_generation                                     = []
        dhw_storage_soc                                      = []
        electrical_storage_soc                               = []
        cooling_demand                                       = []
        dhw_demand                                           = []
        indoor_dry_bulb_temperature_set_point                = []
        
        
        dhw_p = []
        sg_p  = []
        eep_p = []
        cl_p  = []
        
                       
        # Check how many Buildings are in the System!
        for i,b_name in enumerate(self.building_names):            
            
            tmp_observation_names = []
            for obssss in self.observation_names:
                for f_names in obssss: 
                    tmp_observation_names.append(f_names)

            for f_name in list(tmp_observation_names):
                    #for f_name in tmp_observation_names: 
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
                            
                  
            
            
            # Generating the Dataframes per Building
            b_dataframe = pd.DataFrame(columns=['day_type', 'hour', 'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h', 
                                      'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h', 'diffuse_solar_irradiance',
                                      'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 
                                      'diffuse_solar_irradiance_predicted_24h', 'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
                                      'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h', 'carbon_intensity', 
                                      'indoor_dry_bulb_temperature', 'non_shiftable_load', 'solar_generation', 'dhw_storage_soc', 'electrical_storage_soc', 
                                      'electricity_pricing', 'electricity_pricing_predicted_6h', 
                                      'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h', 'cooling_demand',
                                      'dhw_demand','indoor_dry_bulb_temperature_set_point','occupant_count','net_electricity_consumption'])
            
           
            
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
            
            
            b_dataframe = b_dataframe.astype(float)
            
            
            # Making the individual dataframes 
            
            # 1. Cooling Demand
            features = pd.read_csv('my_models/features/feature_importance_cooling_demand.csv')
            b_dataframe_cooling_demand = b_dataframe[features['feature']]
            
            # 2. Carbon Intensity
            features = pd.read_csv('my_models/features/feature_importance_carbon_intensity.csv')
            b_dataframe_carbon_intensity = b_dataframe[features['feature']]
            
            # 3. DHW Demand
            features = pd.read_csv('my_models/features/feature_importance_dhw_demand.csv')
            b_dataframe_dhw_demand = b_dataframe[features['feature']]
            
            # 4. Non Shiftable Load
            features = pd.read_csv('my_models/features/feature_importance_non_shiftable_load.csv')
            b_dataframe_non_shiftable_load = b_dataframe[features['feature']]
            
            # 5. Solar Generation
            features = pd.read_csv('my_models/features/feature_importance_solar_generation.csv')
            b_dataframe_solar_generation = b_dataframe[features['feature']]
              
            if i > 2:
                dhw_p.append(self.dhw_model_list[i-i].predict(b_dataframe))
                sg_p.append(self.sg_model_list[i-i].predict(b_dataframe))
                eep_p.append(self.eep_model_list[i-i].predict(b_dataframe))
                cl_p.append(self.cl_model_list[i-i].predict(b_dataframe))
            else: 
                dhw_p.append(self.dhw_model_list[i].predict(b_dataframe))
                sg_p.append(self.sg_model_list[i].predict(b_dataframe))
                eep_p.append(self.eep_model_list[i].predict(b_dataframe))
                cl_p.append(self.cl_model_list[i].predict(b_dataframe))
        
        sg_total = sg_p[0] + sg_p[1] + sg_p[2]
        cip_p  = self.model_cip.predict(b_dataframe)
        
        
        
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
                            predictions_dict[b_name][load_type] = eep_p[i]
                            
                        if load_type == 'DHW_Heating':
                            predictions_dict[b_name][load_type] = dhw_p[i]
                            
                        if load_type == 'Cooling_Load':
                            predictions_dict[b_name][load_type] = cl_p[i]                       

                predictions_dict['Solar_Generation'] = sg_total
                predictions_dict['Carbon_Intensity'] = cip_p

        self.prev_vals = current_vals
            # ====================================================================
        return predictions_dict
    
    
    
    
    
