from .swin_longnet_encoder import LongNetWithPatchMerging
from torchscale.model import LongNetConfig as longnet_arch
from torchscale.architecture.config import EncoderConfig

def make_longnet(args):
    if args.arch in longnet_arch.__dict__.keys():
        longnet_args = longnet_arch.__dict__[args.arch]
    if hasattr(args, 'dropout'):
        longnet_args['dropout'] = args.dropout
    if hasattr(args, 'drop_path_rate'):
        longnet_args['drop_path_rate'] = args.drop_path_rate
    longnet_args = EncoderConfig(**longnet_args)
    model = LongNetWithPatchMerging(longnet_args)
    print('Number of trainable LongNet parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def make_swin_longnet_from_name(config_name: str,
                           dilated_ratio: str='[1, 2, 4, 8, 16]',
                           segment_length: str='[1024, 2048, 4096, 8192, 16384]',
                           drop_path_rate: int=0.1,
                           dropout: float=0.1,
                           enable_gradient_checkpoint = True,
                           keep_dim_after_merge = True, 
                           merge_size = 2,
                           
                        use_rel_pos_2d = False
                           ):
    '''
    make LongNet model from config name

    Arguments:
    ----------
    config_name: str
        name of the config
    dilated_ratio: str
        dilated ratio
    segment_length: str
        segment length
    drop_path_rate: int
        drop path rate
    dropout: float
        dropout rate
    '''
    if config_name in longnet_arch.__dict__.keys():
        longnet_args = longnet_arch.__dict__[config_name]

    longnet_args = longnet_arch.__dict__[config_name]

    longnet_args['dropout'] = dropout
    longnet_args['drop_path_rate'] = drop_path_rate

    # set dilated ratio and segment length
    longnet_args['dilated_ratio'] = dilated_ratio
    longnet_args['segment_length'] = segment_length
    longnet_args['checkpoint_activations'] = enable_gradient_checkpoint

    print('dilated_ratio: ', dilated_ratio)
    print('segment_length: ', segment_length)

    longnet_args = EncoderConfig(**longnet_args)
    
    model = LongNetWithPatchMerging(longnet_args, 
                                    keep_dim_after_merge=keep_dim_after_merge,
                                    merge_size= merge_size,
                                    use_rel_pos_2d=use_rel_pos_2d
                                    )
    print('Number of trainable LongNet parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

__all__ = [
    'make_swin_longnet_from_name'
]