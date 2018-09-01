import train_config as config
import os
from train import main


    
if __name__ == "__main__":
    temp_c = {}
    original_sub_dir = config.train['sub_dir']
    for i in range(0,326):
        config.train['sub_dir'] = original_sub_dir + '/attribute{}'.format(i) 
        config.train['attribute_index'] = i 
        '''
        for xk,xv in config.__dict__.items():
            if not xk.startswith('_') and isinstance(xv,dict):
                for k in xv:
                    if k in temp_c:
                        getattr(config,xk)[k] = temp_c[k]  
        '''
        print('================================================================================================================================================================================================================================')
        main(config)
