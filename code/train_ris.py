import train_config as config
import os
from train import main


    
if __name__ == "__main__":
    for i in range(0,326):
        config.train['attribute_index'] = i 
        config.parse_config()
        print('================================================================================================================================================================================================================================')
        main(config)
