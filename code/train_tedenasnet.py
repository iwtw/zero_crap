from train import main

if __name__ == '__main__':
    import train_config as config
    config.net['name'] = 'tede_nasnet'
    config.train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
    config.train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
    config.train['log_step'] = 100
    for config.train['optimizer']  in ['SGD' , 'Adam' ]:
        config.parse_config()
        main( config )


