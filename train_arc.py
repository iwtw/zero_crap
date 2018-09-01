import train_config as config
import os
from train import main
from time import sleep


    
if __name__ == "__main__":
    temp_c = {}
    original_sub_dir = config.train['sub_dir']
    for config.loss['weight_mse_attribute'] in [0,1]:
        '''
        for xk,xv in config.__dict__.items():
            if not xk.startswith('_') and isinstance(xv,dict):
                for k in xv:
                    if k in temp_c:
                        getattr(config,xk)[k] = temp_c[k]  
        '''
        print('================================================================================================================================================================================================================================')
        sleep(10)
        try:
            main(config)
        except KeyboardInterrupt:
            pass
