""" utils.py
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def find_best_lr(loss_fn,compute_total_loss_fn,backward_fn,net,optimizer,iter_dataloader,forward_fn,start_lr=1e-5,end_lr = 10 ,num_iters = 100):

    origin_net_state = net.state_dict()
    origin_optimizer_state = optimizer.state_dict()
    best_loss = 1e9
    loss_list = []

    ratio = end_lr / start_lr 
    lr_mult = ratio ** (1/num_iters)
    
    lr_list = [0]
    for it in tqdm(range(num_iters) , desc ='finding lr' ):
        lr = start_lr * lr_mult ** it
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        temp_loss_list = []
        '''
        for it in range(3):
            loss_dict = loss_fn( forward_fn(x,**forward_fn_kwargs) , x ) 
            backward_fn( loss_dict , optimizer )
            with torch.no_grad():
                loss_dict = loss_fn( forward_fn(x,**forward_fn_kwargs) , x ) 
            loss = compute_total_loss_fn( loss_dict ).detach().cpu().numpy()
            temp_loss_list.append( loss )
        '''
        x = next(iter_dataloader)
        for k in x:
            x[k] = x[k].cuda()
            x[k].requires_grad = False
        loss_dict = loss_fn( forward_fn( x ) , x ) 
        loss = compute_total_loss_fn( loss_dict ).detach().cpu().numpy()
        #stop criterion
        if math.isnan( loss ) or loss > 4 * best_loss :
            break

        backward_fn( loss_dict , optimizer )
        lr_list.append( lr )
        loss_list.append( loss )

        #update best_loss
        if it > 10 and loss < best_loss:
            assert num_iters >= 20
            best_loss = loss


    loss_list.append( compute_total_loss_fn( loss_fn( forward_fn(x) , x ) ).detach().cpu().numpy() )


        


    net.load_state_dict( origin_net_state )
    optimizer.load_state_dict( origin_optimizer_state  )

    '''
    steepest_idx = np.argmin(loss_delta_list)
    lowest_idx = np.argmin( loss_list )
    final_lr = lr_list[int(np.floor( steepest_idx + (lowest_idx - steepest_idx) * 3/4   ))  ] 
    tqdm.write('steepest_lr {} lowest_lr {} final_lr {}'.format(lr_list[steepest_idx] , lr_list[lowest_idx] , final_lr))
    return final_lr
    '''
    sm = 0.90
    sm_loss_list = []
    for i in range(len(loss_list)):
        sm_loss_list.append( np.mean( loss_list[max(i-5,0):i+1] ) )
    d_sm_loss_list = [0] + [ sm_loss_list[i] - sm_loss_list[i-1] for i in range(1,len(sm_loss_list)) ]
    dd_sm_loss_list = [0] + [ d_sm_loss_list[i ] - d_sm_loss_list[i-1] for i in range(1,len(d_sm_loss_list)) ]


    plt.figure()
    f , axes = plt.subplots(1,3)
    axes[0].set_title('smoothed loss curve')
    axes[0].set_xlabel('learning rate')
    axes[0].set_ylabel('loss')
    axes[0].plot(  lr_list,sm_loss_list)
    axes[0].set_xscale('log')
    axes[1].set_title('d_loss curve')
    axes[1].set_xlabel('learning rate')
    axes[1].set_ylabel('d_loss')
    axes[1].plot( lr_list,d_sm_loss_list)
    axes[1].set_xscale('log')
    axes[2].set_title('dd_loss curve')
    axes[2].set_xlabel('learning rate')
    axes[2].set_ylabel('ddloss')
    axes[2].plot( lr_list,dd_sm_loss_list)
    axes[2].set_xscale('log')

    steepest_idx = np.argmin( d_sm_loss_list )
    lowest_idx = np.argmin( sm_loss_list )
    final_lr = lr_list[int(np.floor( steepest_idx + (lowest_idx - steepest_idx) * 3/4   ))  ] 
    tqdm.write('steepest_lr {} lowest_lr {} final_lr {}'.format(lr_list[steepest_idx] , lr_list[lowest_idx] , final_lr))
    return final_lr



            





def detect_change(net , mode = 3  ):
    for name, module in net._modules.items():
        grads_list = []
        params_list = []
        for param in module.parameters():
            if param.grad is not None:
                grads_list.append( param.grad.view(-1) )
            params_list.append( param.view(-1) )
        if mode & 1 :
            if len(grads_list) == 0:
                norm = 0
            else:
                norm = torch.norm( torch.cat( grads_list , 0 ) )
            print( "  {} grads norm : {}".format(name,norm) )
        if mode & 2:
            norm = torch.norm( torch.cat( params_list , 0 ) )
            print( '  {} params norm : {}'.format( name , norm) )
        

name_dataparallel = torch.nn.DataParallel.__name__
def lr_warmup(epoch, warmup_length):
    if epoch < warmup_length:
        p = max(0.0, float(epoch)) / float(warmup_length)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0
    torch.nn.PReLU
    

def load_optimizer(optimizer , model , path , epoch = None ):
    """
    return the epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module

    if epoch is None:
        for i in reversed( range(1000) ):
            p = "{}/{}_epoch{}.pth".format( path,type(optimizer).__name__+'_'+type(model).__name__,i )
            if os.path.exists( p ):
                optimizer.load_state_dict(  torch.load( p ) )
                print('Sucessfully resume optimizer {}'.format(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format( path,type(optimizer).__name__+'_'+type(model).__name__,epoch )
        if os.path.exists( p ):
            optimizer.load_state_dict(  torch.load( p )   )
            print('Sucessfully resume optimizer {}'.format(p))
            return epoch
        else:
            warnings.warn("resume optimizer not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1

def load_model(model,path ,epoch = None , strict= True): 
    """
    return the last epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module
    if epoch is None:
        for i in reversed( range(1000) ):
            p = "{}/{}_epoch{}.pth".format( path,type(model).__name__,i )
            if os.path.exists( p ):
                model.load_state_dict(  torch.load( p ) , strict = strict)
                print('Sucessfully resume model {}'.format(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format( path,type(model).__name__,epoch )
        if os.path.exists( p ):
            model.load_state_dict(  torch.load( p ) , strict = strict)
            print('Sucessfully resume model {}'.format(p))
            return epoch
        else:
            warnings.warn("resume model not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1

    
def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b

def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias = drange_out[0]-drange_in[0]*scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
        ])
    return transform(x)


def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


def save_image_single(x, path, imsize=512):
    from PIL import Image
    grid = make_image_grid(x, 1)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)


def save_image_grid(x, path, imsize=512, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)


def save_model(model,dirname,epoch,mode = 'epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch )
    torch.save( model.state_dict() , '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch ) )

def del_model(model,dirname,epoch,mode = 'epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch )
    if os.path.exists( model_pathname ):
        os.system('rm {}'.format(model_pathname))

def save_optimizer(optimizer,model,dirname,epoch,mode='epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save( optimizer.state_dict() , '{}/{}_epoch{}.pth'.format(dirname,type(optimizer).__name__ +'_' +type(model).__name__,epoch ) )

def del_optimizer(optimizer,model,dirname,epoch,mode='epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_epoch{}.pth'.format(dirname,type(optimizer).__name__ +'_' +type(model).__name__,epoch )
    if os.path.exists( model_pathname ):
        os.system('rm {}'.format(model_pathname))


def make_summary(writer, key, value, step):
    if hasattr(value, '__len__'):
        for idx, img in enumerate(value):
            summary = tf.Summary()
            sio = BytesIO()
            scipy.misc.toimage(img).save(sio, format='png')
            image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
            summary.value.add(tag="{}/{}".format(key, idx), image=image_summary)
            writer.add_summary(summary, global_step=step)
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, global_step=step)




import torch
import math
irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                #for pytorch 0.3.0
                #norm_ip(t, t.min(), t.max())
                #for pytorch 0.4.0
                norm_ip(t, float(t.min()) , float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
