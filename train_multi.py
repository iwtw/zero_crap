import train_config as config
import os
from train import main


    
if __name__ == "__main__":
    temp_c = {}
    for temp_c['learning_rate'] in [1e-2]:
        for temp_c['first_kernel_size'] in [7]:
            for temp_c['use_maxpool'],temp_c['strides'],RF in zip([True,False,False,False],[[1,2,2,2],[1,2,2,2],[2,2,1,1],[1,1,2,2]],[435,219,171,123]):#RF:219, 171 , 123
                config.train['sub_dir'] = '{}-first_kernel_size{}-{}-m{:.1f}-s{}-{}'.format( config.train['dataset'], temp_c['first_kernel_size'] ,  'maxpool' if temp_c['use_maxpool'] else 'nomaxpool' , config.loss['m'] , config.loss['s'],  config.train['optimizer'] )
                for xk,xv in config.__dict__.items():
                    if not xk.startswith('_') and isinstance(xv,dict):
                        for k in xv:
                            if k in temp_c:
                                getattr(config,xk)[k] = temp_c[k]  
                print('================================================================================================================================================================================================================================')
                print('Config :')
                for k,v in config.__dict__.items():
                    if not k.startswith('_'):
                        print("{} = {}".format(k,v))
                print("")
                main(config)


