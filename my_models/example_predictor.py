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

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss

from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
import re

from my_models.base_predictor_model import BasePredictorModel


class ExamplePredictor(BasePredictorModel):

    
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
        print("=========================Available Observations=========================")

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
        print("Loading the Models!")
        self.model_dhw_b1 = lgb.Booster(model_file='my_models/models/dhw_load_model_b1.txt')
        self.model_dhw_b2 = lgb.Booster(model_file='my_models/models/dhw_load_model_b2.txt')
        self.model_dhw_b3 = lgb.Booster(model_file='my_models/models/dhw_load_model_b3.txt')
        self.model_sg_b1  = lgb.Booster(model_file='my_models/models/solar_generation_model_b1.txt')
        self.model_sg_b2  = lgb.Booster(model_file='my_models/models/solar_generation_model_b2.txt')
        self.model_sg_b3  = lgb.Booster(model_file='my_models/models/solar_generation_model_b3.txt')
        self.model_eep_b1 = lgb.Booster(model_file='my_models/models/Equipment_Electric_Power_model_b1.txt')
        self.model_eep_b2 = lgb.Booster(model_file='my_models/models/Equipment_Electric_Power_model_b2.txt')
        self.model_eep_b3 = lgb.Booster(model_file='my_models/models/Equipment_Electric_Power_model_b3.txt')
        self.model_cl_b1  = lgb.Booster(model_file='my_models/models/cooling_load_model_b1.txt')
        self.model_cl_b2  = lgb.Booster(model_file='my_models/models/cooling_load_model_b1.txt')
        self.model_cl_b3  = lgb.Booster(model_file='my_models/models/cooling_load_model_b1.txt')
        self.model_cip    = lgb.Booster(model_file='my_models/models/Carbon_Intensity_Power_model.txt')
        print("Finished Loading the Models")

    def compute_forecast(self, observations):
        """Compute forecasts for each variable given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation. The structure of this list can
                be viewed via CityLearnEnv.observation_names.

        Returns:
            predictions_dict (dict): dictionary containing forecasts for each
                variable. Format is as follows:
                {
                    'Building_1': { # this is env.buildings[0].name
                        'Equipment_Eletric_Power': [ list of 48 floats - predicted equipment electric power for Building_1 ],
                        'DHW_Heating': [ list of 48 floats - predicted DHW heating for Building_1 ],
                        'Cooling_Load': [ list of 48 floats - predicted cooling load for Building_1 ]
                        },
                    'Building_2': ... (as above),
                    'Building_3': ... (as above),
                    'Solar_Generation': [ list of 48 floats - predicted solar generation ],
                    'Carbon_Intensity': [ list of 48 floats - predicted carbon intensity ]
                }
        """

        # ====================================================================
        # insert your forecasting code here
        # ====================================================================
        
        
        # Take all the input features        
        
        # Result
        dhw_1_p = []
        sg_1_p  = []
        eep_1_p = []
        cl_1_p  = []
        
        dhw_2_p = []
        sg_2_p  = []
        eep_2_p = []
        cl_2_p  = []
        
        dhw_3_p = []
        sg_3_p  = []
        eep_3_p = []
        cl_3_p  = []
        
        dhw_4_p = []
        sg_4_p  = []
        eep_4_p = []
        cl_4_p  = []
        
        dhw_5_p = []
        sg_5_p  = []
        eep_5_p = []
        cl_5_p  = []
        
        dhw_6_p = []
        sg_6_p  = []
        eep_6_p = []
        cl_6_p  = []
        
        
        ### ENV Setting
        env = 'local'
        
        if env == 'local':
            print("Im working locally!")
        if env == 'online':
            print("Im working online!")
        
        if env == 'local': 
            # 1. Housing Level
            for i,b_name in enumerate(self.building_names):
            
                print("Housing Number: " + str(i))

                v_list = []
                f_list = []
                for obs in self.observation_names:
                    for obss in obs: 
                        tmp = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == obss)[0][0]])
                        v_list.append(tmp)
                        f_list.append(obss)

            
                df = pd.DataFrame(columns=f_list)
                i = 0
                for (columnName, columnData) in df.iteritems():
                    df.at[0, columnName] = v_list[i]
                    i = i + 1
                df = df.astype(float)
       

                if b_name == 'Building_1':
                    dhw_1_p = self.model_dhw_b1.predict(df,predict_disable_shape_check=True)
                    sg_1_p  = self.model_sg_b1.predict(df,predict_disable_shape_check=True)
                    eep_1_p = self.model_eep_b1.predict(df,predict_disable_shape_check=True)
                    cl_1_p  = self.model_cl_b1.predict(df,predict_disable_shape_check=True)
       
                if b_name == 'Building_2':
                    dhw_2_p = self.model_dhw_b2.predict(df,predict_disable_shape_check=True)
                    sg_2_p  = self.model_sg_b2.predict(df,predict_disable_shape_check=True)
                    eep_2_p = self.model_eep_b2.predict(df,predict_disable_shape_check=True)
                    cl_2_p  = self.model_cl_b2.predict(df,predict_disable_shape_check=True)
                
                if b_name == 'Building_3':
                    dhw_3_p = self.model_dhw_b3.predict(df,predict_disable_shape_check=True)
                    sg_3_p  = self.model_sg_b3.predict(df,predict_disable_shape_check=True)
                    eep_3_p = self.model_eep_b3.predict(df,predict_disable_shape_check=True)
                    cl_3_p  = self.model_cl_b3.predict(df,predict_disable_shape_check=True)

                
            # 2. Neighbourhood Level
            sg_total = sg_1_p + sg_2_p + sg_3_p
            cip_p  = self.model_cip.predict(df,predict_disable_shape_check=True)

        
            print("Setting the current Values")
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
        

            print("Setting the Predictions")
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

                for b_name in self.building_names:
                    predictions_dict[b_name] = {}
                
                    for load_type in ['Equipment_Eletric_Power','DHW_Heating','Cooling_Load']:
                    
                        if load_type == 'Equipment_Eletric_Power':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = eep_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = eep_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = eep_3_p
                            
                        if load_type == 'DHW_Heating':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = dhw_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = dhw_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = dhw_3_p
                            
                        if load_type == 'Cooling_Load':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = cl_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = cl_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = cl_3_p                          

                predictions_dict['Solar_Generation'] = sg_total
                predictions_dict['Carbon_Intensity'] = cip_p

            self.prev_vals = current_vals
            # ====================================================================
            print("Done Prediction!")
            
        if env == 'online': 
            # 1. Housing Level
            for i,b_name in enumerate(self.building_names):
                print("Housing Number: " + str(i))
            
                v_list = []
                f_list = []
                for obs in self.observation_names:
                    for obss in obs: 
                        tmp = (np.array(observations)[0][np.where(np.array(self.observation_names)[0] == obss)[0][0]])
                        v_list.append(tmp)
                        f_list.append(obss)

            
                df = pd.DataFrame(columns=f_list)
                i = 0
                for (columnName, columnData) in df.iteritems():
                    df.at[0, columnName] = v_list[i]
                    i = i + 1
                df = df.astype(float)
    
                if b_name == 'Building_1':
                    dhw_1_p = self.model_dhw_b1.predict(df,predict_disable_shape_check=True)
                    sg_1_p  = self.model_sg_b1.predict(df,predict_disable_shape_check=True)
                    eep_1_p = self.model_eep_b1.predict(df,predict_disable_shape_check=True)
                    cl_1_p  = self.model_cl_b1.predict(df,predict_disable_shape_check=True)
       
                if b_name == 'Building_2':
                    dhw_2_p = self.model_dhw_b2.predict(df,predict_disable_shape_check=True)
                    sg_2_p  = self.model_sg_b2.predict(df,predict_disable_shape_check=True)
                    eep_2_p = self.model_eep_b2.predict(df,predict_disable_shape_check=True)
                    cl_2_p  = self.model_cl_b2.predict(df,predict_disable_shape_check=True)
                
                if b_name == 'Building_3':
                    dhw_3_p = self.model_dhw_b3.predict(df,predict_disable_shape_check=True)
                    sg_3_p  = self.model_sg_b3.predict(df,predict_disable_shape_check=True)
                    eep_3_p = self.model_eep_b3.predict(df,predict_disable_shape_check=True)
                    cl_3_p  = self.model_cl_b3.predict(df,predict_disable_shape_check=True)
                    
                if b_name == 'Building_4':
                    dhw_4_p = self.model_dhw_b4.predict(df,predict_disable_shape_check=True)
                    sg_4_p  = self.model_sg_b4.predict(df,predict_disable_shape_check=True)
                    eep_4_p = self.model_eep_b4.predict(df,predict_disable_shape_check=True)
                    cl_4_p  = self.model_cl_b4.predict(df,predict_disable_shape_check=True)
                    
                if b_name == 'Building_5':
                    dhw_5_p = self.model_dhw_b5.predict(df,predict_disable_shape_check=True)
                    sg_5_p  = self.model_sg_b5.predict(df,predict_disable_shape_check=True)
                    eep_5_p = self.model_eep_b5.predict(df,predict_disable_shape_check=True)
                    cl_5_p  = self.model_cl_b5.predict(df,predict_disable_shape_check=True)
                    
                if b_name == 'Building_6':
                    dhw_6_p = self.model_dhw_b6.predict(df,predict_disable_shape_check=True)
                    sg_6_p  = self.model_sg_b6.predict(df,predict_disable_shape_check=True)
                    eep_6_p = self.model_eep_b6.predict(df,predict_disable_shape_check=True)
                    cl_6_p  = self.model_cl_b6.predict(df,predict_disable_shape_check=True)

                
            # 2. Neighbourhood Level
            sg_total = sg_1_p + sg_2_p + sg_3_p + sg_4_p + sg_5_p + sg_6_p
            cip_p  = self.model_cip.predict(df,predict_disable_shape_check=True)

        
            print("Setting the current Values")
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
        

            print("Setting the Predictions")
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

                for b_name in self.building_names:
                    predictions_dict[b_name] = {}
                
                    for load_type in ['Equipment_Eletric_Power','DHW_Heating','Cooling_Load']:
                    
                        if load_type == 'Equipment_Eletric_Power':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = eep_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = eep_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = eep_3_p
                            if b_name == 'Building_4':
                                predictions_dict[b_name][load_type] = eep_4_p
                            if b_name == 'Building_5':
                                predictions_dict[b_name][load_type] = eep_5_p
                            if b_name == 'Building_6':
                                predictions_dict[b_name][load_type] = eep_6_p
                            
                        if load_type == 'DHW_Heating':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = dhw_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = dhw_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = dhw_3_p
                            if b_name == 'Building_4':
                                predictions_dict[b_name][load_type] = dhw_4_p
                            if b_name == 'Building_5':
                                predictions_dict[b_name][load_type] = dhw_5_p
                            if b_name == 'Building_6':
                                predictions_dict[b_name][load_type] = dhw_6_p
                            
                        if load_type == 'Cooling_Load':
                            if b_name == 'Building_1':
                                predictions_dict[b_name][load_type] = cl_1_p
                            if b_name == 'Building_2':
                                predictions_dict[b_name][load_type] = cl_2_p
                            if b_name == 'Building_3':
                                predictions_dict[b_name][load_type] = cl_3_p     
                            if b_name == 'Building_4':
                                predictions_dict[b_name][load_type] = cl_4_p
                            if b_name == 'Building_5':
                                predictions_dict[b_name][load_type] = cl_5_p
                            if b_name == 'Building_6':
                                predictions_dict[b_name][load_type] = cl_6_p     

                predictions_dict['Solar_Generation'] = sg_total
                predictions_dict['Carbon_Intensity'] = cip_p
                
            self.prev_vals = current_vals
            # ====================================================================
        print("Done Prediction!")
        print("Prediction Solar: " + str(predictions_dict['Solar_Generation']))
        return predictions_dict
    
    
    
    
    
