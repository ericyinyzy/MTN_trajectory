import copy
import os
import numpy as np
import scipy.io
import scipy.spatial
import torch
from torch.utils.data import Dataset
from jaad_data import JAAD
def generate_mean_std():
    data_path = 'JAAD_dataset'
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'subset': 'high_visibility',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}
    imdb = JAAD(data_path=data_path)
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    tracks=[]
    for track in beh_seq_train['bbox']:
        tracks.extend([track[i:i + 60] for i in
                       range(0, len(track) - 60 + 1, 7)])
    trac = np.array(tracks).reshape(-1, 4)
    mean = trac.mean(0)
    std = trac.std(0)
    return mean,std


def create_jaad_dataset(mean,std,dataset='jaad',flag='train',**model_opts):

    data_path = 'JAAD_dataset'
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'subset': 'high_visibility',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                   'val_data': True,
                                   'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}
    imdb = JAAD(data_path=data_path)
    data_list={}

    if flag=='train':
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
        beh_seq_train['ego_op_flow'] =np.load('flow/flow_JAAD_train_ego.npy',allow_pickle=True)
        beh_seq_train['ped_op_flow']=np.load('flow/flow_JAAD_train_ped.npy',allow_pickle=True)
        data_list= get_data(beh_seq_train,'train',mean,std,**model_opts)
    elif flag=='val':
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
        beh_seq_val['ego_op_flow'] = np.load('flow/flow_JAAD_val_ego.npy', allow_pickle=True)
        beh_seq_val['ped_op_flow'] = np.load('flow/flow_JAAD_val_ped.npy', allow_pickle=True)
        data_list= get_data(beh_seq_val,'val',mean,std,**model_opts)
    elif flag=='test':
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
        beh_seq_test['ego_op_flow'] = np.load('flow/flow_JAAD_test_ego.npy', allow_pickle=True)
        beh_seq_test['ped_op_flow'] =np.load('flow/flow_JAAD_test_ped.npy', allow_pickle=True)
        data_list= get_data(beh_seq_test, 'test',mean,std,**model_opts)
    data={}
    data['image_name'] = data_list['obs_image']
    data['pid']=data_list['obs_pid']
    data['ego_op_flow']=data_list['ego_op_flow']
    data['ped_op_flow']=data_list['ped_op_flow']
    data['enc_input']=data_list['enc_input']
    data['dec_input']=data_list['dec_input']
    data['pred_target']=data_list['pred_target']
    return OnboardTfDataset(data, flag, mean, std)

def get_tracks(dataset, data_types, observe_length,dataset_type, predict_length, overlap, normalize,mean,std):
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
    #  Calculates the overlap in terms of number of frames
    seq_length = observe_length + predict_length
    # print('length_set',seq_length,observe_length,predict_length)
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


    for k in d.keys():     #'image',pid,bbox op_flos ped_op_flow
        tracks = []
        for track in d[k]:
            tracks.extend([track[i:i + seq_length] for i in
                           range(0, len(track) - seq_length + 1, overlap_stride)])
        d[k] = tracks
    if 'bbox' in data_types:
        box=copy.deepcopy(d['bbox'])
    else:
        box=None
    d['scale']=[]
    if normalize:
        if 'bbox' in data_types:
            for i in range(len(d['bbox'])):
                d['bbox'][i] = np.divide(np.subtract(d['bbox'][i], mean),std)
        if 'center' in data_types:
            for i in range(len(d['center'])):
                d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
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
        return np.concatenate(d, axis=2)
    else:
        return d[0]

def get_data(data,flag,mean,std,**model_opts):
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
                                  opts['normalize_bbox'],mean,std)
    scale=np.array(data_tracks['scale'])
    obs_slices = {}
    pred_slices = {}
    all_slices={}
    if opts['enc_input_type']==['obd_speed']:
        obs_box=None
        pred_box=None
    else:
        obs_box = []
        obs_box.extend([d[:observe_length ] for d in box_viz])
        pred_box = []
        pred_box.extend([d[observe_length:] for d in box_viz])
    for k in data_tracks.keys():
        obs_slices[k] = []
        pred_slices[k] = []
        obs_slices[k].extend([d[:observe_length] for d in data_tracks[k]])
        if k=='obd_speed':
            pred_slices[k].extend([d for d in data_tracks[k]])
        else:
            pred_slices[k].extend([d[observe_length:] for d in data_tracks[k]])
    all_slices['image']=[]
    all_slices['image'].extend([d[0:] for d in data_tracks['image']])
    all_slices['pid']=[]
    all_slices['pid'].extend([d[0:] for d in data_tracks['pid']])
    all_slices['ego_op_flow']=[]
    all_slices['ego_op_flow'].extend(d[0:] for d in data_tracks['ego_op_flow'])
    all_slices['ped_op_flow'] = []
    all_slices['ped_op_flow'].extend(d[0:] for d in data_tracks['ped_op_flow'])
    all_slices['bbox']=[]
    all_slices['bbox'].extend(d[0:] for d in data_tracks['bbox'])
    enc_input = get_data_helper(obs_slices, opts['enc_input_type'])
    type=[]
    dec_input = get_data_helper(pred_slices, type)
    pred_target = get_data_helper(pred_slices, opts['prediction_type'])
    if not len(dec_input) > 0:
        dec_input = np.zeros(shape=pred_target.shape)
    return {'obs_image': obs_slices['image'],
            'obs_pid': obs_slices['pid'],
            'all_image':all_slices['image'],
            'all_pid': all_slices['pid'],
            'pred_image': pred_slices['image'],
            'pred_pid': pred_slices['pid'],
            'ego_op_flow':all_slices['ego_op_flow'],
            'ped_op_flow': all_slices['ped_op_flow'],
            'enc_input': enc_input,
            'dec_input': dec_input,
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
        input=self.data['enc_input'][index]
        fp = input[-1, :]
        velocity = (np.diff(input, axis=0).sum(axis=0)) / 14
        step = np.arange(1, 46)
        future_xtl = fp[0] + step * velocity[0]
        future_ytl = fp[1] + step * velocity[1]
        future_xbr = fp[2] + step * velocity[2]
        future_ybr = fp[3] + step * velocity[3]
        constant_prediction = np.stack((future_xtl, future_ytl, future_xbr, future_ybr), axis=-1)

        return {'enc_input': torch.Tensor(self.data['enc_input'][index]),
                'dec_input': torch.Tensor(self.data['dec_input'][index]),
                'linear_traj':torch.Tensor(constant_prediction),
                'ego_op_flow':torch.Tensor(self.data['ego_op_flow'][index]),
                'ped_op_flow': torch.Tensor(self.data['ped_op_flow'][index]),
                'pid': self.data['pid'][index][0][0],
                'image_name': self.data['image_name'][index][0],
                'pred_target': self.data['pred_target'][index],
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



def get_strided_data(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_no_start = inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm=inp_no_start

    return inp_norm[:,:gt_size-1],inp_norm[:,gt_size-1:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors
