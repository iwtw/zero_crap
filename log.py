##################################################
#borrowed from https://github.com/nashory/pggan-pytorch
##################################################
import torch
import numpy as np
import torchvision.models as models
import utils as utils
from tensorboardX import SummaryWriter
import os, sys
import importlib
from types import ModuleType
import datetime
from tqdm import tqdm


class TensorBoardX:
    def __init__(self,config , sub_dir ="" , log_type = ['train','val'] ):
        if sub_dir != "":
            sub_dir = '/' + sub_dir
        os.system('mkdir -p save{}'.format(sub_dir))
        for i in range(1000):
            self.path = 'save{}/{}'.format(sub_dir , i)
            if not os.path.exists(self.path):
                break
        os.system('mkdir -p {}'.format(self.path))
        print("Saving logs at {}".format(self.path))
        self.writer = {}
        for k in log_type:
            self.writer[k] = SummaryWriter( self.path +'/' + k )
            self.writer[k] = SummaryWriter( self.path +'/' + k )


        #Export run arguments
        with open(os.path.join(self.path, 'run.txt'), 'wt') as f:
            f.write('%-16s%s\n' % ('Date', datetime.datetime.today()))
            f.write('%-16s%s\n' % ('Working dir', os.getcwd()))
            f.write('%-16s%s\n' % ('Executable', sys.argv[0]))
            f.write('%-16s%s\n' % ('Arguments', ' '.join(sys.argv[1:])))
        #Export config
        with open(os.path.join(self.path, 'config.txt'), 'wt') as fout:
            for k, v in sorted(config.__dict__.items()):
                if not k.startswith('_') and not isinstance( v , ModuleType ) :
                    fout.write("%s = %s\n" % (k, str(v)))
        self.logger = open( os.path.join(self.path,'log.txt') ,'w' )
        self.err_logger = open( os.path.join(self.path,'err.txt') ,'w' )

        #os.system('cp {} {}/'.format(config_filename , self.path))
                
    def add_scalar(self, index, val, niter , logtype):
        self.writer[logtype].add_scalar(index, val, niter)

    def add_scalars(self, index, group_dict, niter , logtype):
        self.writer[logtype].add_scalar(index, group_dict, niter)

    def add_image_grid(self, index, ngrid, x, niter , logtype):
        grid = utils.make_image_grid(x, ngrid)
        self.writer[logtype].add_image(index, grid, niter)

    def add_image_single(self, index, x, niter , logtype):
        self.writer[logtype].add_image(index, x, niter)
    def add_histogram(self, index , x , niter , logtype):
        self.writer[logtype].add_histogram( index , x , niter )


    def add_graph(self, index, x_input, model , logtype):
        torch.onnx.export(model, x_input, os.path.join(self.path, "{}.proto".format(index)), verbose=True)
        self.writer[logtype].add_graph_onnx(os.path.join(self.path, "{}.proto".format(index)))

    def export_json(self, out_file , logtype ):
        self.writer[logtype].export_scalars_to_json(out_file)
    def write_log(self , msg , end = '\n' , use_tqdm = True ):
        if use_tqdm:
            tqdm.write(msg, file=sys.stdout, end=end)
        else:
            print(msg,end=end,file=sys.stdout)
        self.logger.write( msg + end )
        sys.stdout.flush()
        self.logger.flush()
    def write_err(self , msg , end = '\n' ):
        sys.stderr.write( msg , end = end )
        self.err_logger.write( msg + end )
        sys.stderr.flush()
        self.err_logger.flush()
    def write_net(self , msg ):
        with open(os.path.join(self.path,'net.txt') , 'w') as fp:
            fp.write( msg +'\n' )
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        

        





