from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import numpy as np


class log_tensorboard_data:
    """
    SUMMARY: This class is for parsing tensorboard data from a given path.


    """
    def __init__(self):
        pass


    def parse_tensorboard(self, path, reward_dict):
        """returns a dictionary of pandas dataframes for each requested scalar"""
        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        _absorb_print = ea.Reload()
        print(ea.Tags()["scalars"])
        # make sure the scalars are in the event accumulator tags

        if not(reward_dict in ea.Tags()["scalars"]):
            print("WARNING: Reward Total is not yet calculated. using instantanous reward instead")
            
        reward_dict = 'Reward / Instantaneous reward (mean)'


        
        return {reward_dict: pd.DataFrame(ea.Scalars(reward_dict))}


    def get_tensorboard_data(self, path):

        reward_dict = ['Reward / Total reward (mean)']

        data = self.parse_tensorboard(path, reward_dict)
        
        
        reward_val = np.array(data['Reward / Total reward (mean)'].get('value'))
        reward_val_mean = np.mean(reward_val)
        reward_val_std = np.std(reward_val)

        return reward_val_mean, reward_val_std



