import copy
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from pie_data import PIE

def generate_mean_std():
    data_path = 'PIE_dataset'
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}
    imdb = PIE(data_path=data_path)
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    tracks=[]
    tracks_spd=[]
    for track in beh_seq_train['bbox']:
        tracks.extend([track[i:i + 60] for i in
                       range(0, len(track) - 60 + 1, 7)])
    for track in beh_seq_train['obd_speed']:
        tracks_spd.extend([track[i:i + 60] for i in
                       range(0, len(track) - 60 + 1, 7)])
    trac = np.array(tracks).reshape(-1, 4)
    mean = trac.mean(0)
    std = trac.std(0)
    trac1 = np.array(tracks_spd).reshape(-1, 1)
    mean_speed = trac1.mean(0)
    std_speed = trac1.std(0)
    return mean,std,mean_speed,std_speed

def create_pie_dataset(mean,std,mean_speed,std_speed,dataset='pie',flag='train',**model_opts):
    pie_path = 'PIE_dataset'
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}

    if dataset == 'pie':
        imdb = PIE(data_path=pie_path)
    if flag=='train':
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_train['ego_op_flow'] =np.load('flow/flow_PIE_train_ego.npy',allow_pickle=True)
        beh_seq_train['ped_op_flow']=np.load('flow/flow_PIE_train_ped.npy',allow_pickle=True)
        data_list = get_data(beh_seq_train,'train',mean,std,mean_speed,std_speed,**model_opts)
    elif flag=='val':
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_val['ego_op_flow'] = np.load('flow/flow_PIE_val_ego.npy',allow_pickle=True)
        beh_seq_val['ped_op_flow'] = np.load('flow/flow_PIE_val_ped.npy', allow_pickle=True)
        data_list = get_data(beh_seq_val,'val',mean,std,mean_speed,std_speed,**model_opts)
    elif flag=='test':
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        beh_seq_test['ego_op_flow'] = np.load('flow/flow_PIE_test_ego.npy',allow_pickle=True)
        beh_seq_test['ped_op_flow'] = np.load('flow/flow_PIE_test_ped.npy', allow_pickle=True)
        data_list = get_data(beh_seq_test, 'test',mean,std,mean_speed,std_speed,**model_opts)
    data={}
    data['image_name'] = data_list['obs_image']
    data['ego_op_flow']=data_list['ego_op_flow']
    data['enc_input']=data_list['enc_input']
    data['obd_speed']=data_list['obd_speed']
    data['ped_op_flow']=data_list['ped_op_flow']
    data['pred_target']=data_list['pred_target']
    return OnboardTfDataset(data, flag, mean, std)

def get_tracks(dataset, data_types, observe_length,dataset_type, predict_length, overlap, normalize,mean,std,mean_speed,std_speed):
    """
    Generates tracks by sampling from pedestrian sequences
    :param dataset: The raw data passed to the method
    :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
    JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
    :param observe_length: The length of the observation (i.e. time steps of the encoder)
    :param predict_length: The length of the prediction (i.e. time steps of the decoder)
    :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
    :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
    the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
    :return: A dictinary containing sampled tracks for each data modality
    """
    seq_length = observe_length + predict_length
    overlap_stride = observe_length if overlap == 0 else \
        int((1 - overlap) * observe_length)
    overlap_stride = 1 if overlap_stride < 1 else overlap_stride
    d = {}

    for dt in data_types:
        print('data_type',dt)
        try:
            d[dt] = dataset[dt]
        except KeyError:
            raise ('Wrong data type is selected %s' % dt)

    d['image'] = dataset['image']
    d['pid'] = dataset['pid']
    for k in d.keys():     #'image',pid,bbox
        tracks = []
        for track in d[k]:
            tracks.extend([track[i:i + seq_length] for i in
                           range(0, len(track) - seq_length + 1, overlap_stride)])
        d[k] = tracks
    print('data_types',data_types)
    if dataset_type=='train' and 'bbox' in data_types:
        trac=np.array(d['bbox']).reshape(-1, 4)
        mean = trac.mean(0)
        std = trac.std(0)
    if dataset_type=='train' and 'obd_speed' in data_types:
        trac1 = np.array(d['obd_speed']).reshape(-1, 1)
        mean_speed=trac1.mean(0)
        std_speed=trac1.std(0)
    if 'bbox' in data_types:
        box=copy.deepcopy(d['bbox'])
    else:
        box=None
    d['scale']=[]

    if normalize:
        if 'bbox' in data_types:
            for i in range(len(d['bbox'])):
                d['bbox'][i] = np.divide(np.subtract(d['bbox'][i], mean),std)
        if 'obd_speed' in data_types:
            for i in range(len(d['obd_speed'])):
                d['obd_speed'][i] = np.divide(np.subtract(d['obd_speed'][i], mean_speed),std_speed).tolist()
        if 'center' in data_types:
            for i in range(len(d['center'])):
                d['center'][i] = np.subtract(d['center'][i], d['center'][i][0]).tolist()
        #  Adjusting the length of other data types
        for k in d.keys():
            if k != 'bbox' and k != 'center'and k!='scale' and k!='obd_speed' and k!='ego_op_flow' and k!='ped_op_flow':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i]

    return d,box

def get_data_helper(data, data_type):
    """
    A helper function for data generation that combines different data types into a single representation
    :param data: A dictionary of different data types
    :param data_type: The data types defined for encoder and decoder input/output
    :return: A unified data representation as a list
    """
    if not data_type:
        return []
    d = []
    for dt in data_type:
        if dt == 'image':
            continue
        d.append(np.array(data[dt]))
    if len(d) > 1:
        for i in d:
            print('sss',i.shape)
        return np.concatenate(d, axis=2)
    else:
        return d[0]

def get_data(data,flag,mean,std,mean_speed,std_speed,**model_opts):
    """
    Main data generation function for training/testing
    :param data: The raw data
    :param model_opts: Control parameters for data generation characteristics (see below for default values)
    :return: A dictionary containing training and testing data
    """
    opts = {
        'normalize_bbox': True,
        'track_overlap': 0.5,
        'observe_length': 15,
        'predict_length': 45,
        'enc_input_type': ['bbox'],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }

    for key, value in model_opts.items():
        assert key in opts.keys(), 'wrong data parameter %s' % key
        opts[key] = value

    observe_length = opts['observe_length']
    data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
    data_tracks,box_viz = get_tracks(data, data_types, observe_length,flag,
                                  opts['predict_length'], opts['track_overlap'],
                                  opts['normalize_bbox'],mean,std,mean_speed,std_speed)

    scale=np.array(data_tracks['scale'])
    obs_slices = {}
    pred_slices = {}
    all_slices={}

    if opts['enc_input_type']==['obd_speed']:
        obs_box=None
        pred_box=None
        all_box=None
    else:
        obs_box = []
        all_box=[]
        all_box.extend([d[1:] for d in box_viz])
        obs_box.extend([d[:observe_length] for d in box_viz])
        pred_box = []
        pred_box.extend([d[observe_length:] for d in box_viz])
    for k in data_tracks.keys():
        obs_slices[k] = []
        pred_slices[k] = []
        obs_slices[k].extend([d[0:observe_length] for d in data_tracks[k]])
        if k=='obd_speed':
            pred_slices[k].extend([d for d in data_tracks[k]])
        else:
            pred_slices[k].extend([d[observe_length:] for d in data_tracks[k]])

    all_slices['image']=[]
    all_slices['image'].extend([d[0:] for d in data_tracks['image']])
    all_slices['pid']=[]
    all_slices['ego_op_flow']=[]
    all_slices['ego_op_flow'].extend(d[0:] for d in data_tracks['ego_op_flow'])
    all_slices['ped_op_flow'] = []
    all_slices['ped_op_flow'].extend(d[0:] for d in data_tracks['ped_op_flow'])
    all_slices['pid'].extend([d[0:] for d in data_tracks['pid']])
    all_slices['bbox']=[]
    all_slices['bbox'].extend(d[0:] for d in data_tracks['bbox'])
    enc_input = get_data_helper(obs_slices, opts['enc_input_type'])
    type=['obd_speed']
    obd_speed = get_data_helper(pred_slices, type)
    pred_target = get_data_helper(pred_slices, opts['prediction_type'])
    if not len(obd_speed) > 0:
        obd_speed = np.zeros(shape=pred_target.shape)
    return {'obs_image': obs_slices['image'],
            'obs_pid': obs_slices['pid'],
            'all_image':all_slices['image'],
            'all_pid': all_slices['pid'],
            'pred_image': pred_slices['image'],
            'pred_pid': pred_slices['pid'],
            'ego_op_flow':all_slices['ego_op_flow'],
            'ped_op_flow': all_slices['ped_op_flow'],
            'enc_input': enc_input,
            'obd_speed': obd_speed,
            'pred_target': pred_target,
            'model_opts': opts,
            'obs_box':obs_box,
            'all_bbox':all_slices['bbox'],
            'pred_box':pred_box,
            'scale':scale}

class OnboardTfDataset(Dataset):
    def __init__(self,data,name,mean,std):
        super(OnboardTfDataset,self).__init__()

        self.data=data
        self.name=name
        self.mean=mean
        self.std=std

    def __len__(self):
        return self.data['enc_input'].shape[0]

    # ef
    def __getitem__(self, index):
        return {'enc_input': torch.Tensor(self.data['enc_input'][index]),
                'obd_speed': torch.Tensor(self.data['obd_speed'][index]),
                'ego_op_flow':torch.Tensor(self.data['ego_op_flow'][index]),
                'ped_op_flow': torch.Tensor(self.data['ped_op_flow'][index]),
                'image_name': self.data['image_name'][index][0],
                'pred_target': torch.Tensor(self.data['pred_target'][index]),
                }


def create_folders(baseFolder,datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder,datasetName))
    except:
        pass
