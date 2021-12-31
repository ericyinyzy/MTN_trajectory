import argparse
import baselineUtils
import torch
import torch.utils.data
import os
import individual_TF
import numpy as np
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(model_path):
    traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': 45,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': ['ped_op_flow','ego_op_flow'],
                       'prediction_type': ['bbox']
                       }
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--model_opts', type=dict, default=traj_model_opts)
    parser.add_argument('--emb_size',type=int,default=32)
    parser.add_argument('--ff_size', type=int, default=128)
    parser.add_argument('--heads',type=int, default=4)
    parser.add_argument('--layers',type=int,default=3)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--input_length', type=int, default=15)
    parser.add_argument('--ego_patches', type=int, default=64)
    parser.add_argument('--ped_patches', type=int, default=9)
    args = parser.parse_args()
    inp_l = args.input_length
    ego_patches = args.ego_patches
    ped_patches = args.ped_patches
    device=torch.device("cuda")
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    mean_tr,std_tr=baselineUtils.generate_mean_std()
    test_dataset=baselineUtils.create_jaad_dataset(mean_tr,std_tr,dataset='jaad',flag='test',**args.model_opts)
    model = individual_TF.IndividualTF(inp_l,4, 4, 4, N=args.layers,
                                       d_model=args.emb_size, d_ff=args.ff_size, h=args.heads, dropout=args.dropout).to(device)
    total_params=0
    for params in model.parameters():
        num_params=1
        for x in params.size():
            num_params *= x
        total_params += num_params
    print("total_parameters: {}".format(total_params))       
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    def split_resultbytime(test_result):
        for i in range(45):
            print('time:{},CMSE:{}'.format(i+1, test_result[:, i, :].mean(axis=None)))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    with torch.no_grad():
        model.eval()
        mse_batch_list = []
        mse_batch_center_list=[]
        for  batch in tqdm(test_dl):
            inp = batch['enc_input'].float().to(device)
            ego_op_flow = batch['ego_op_flow'].permute(0, 1, 3, 2)
            enc_extra_ego = ego_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
            ped_op_flow = batch['ped_op_flow'].permute(0, 1, 3, 2)
            enc_extra_ped = ped_op_flow[:, :(inp_l-1)].reshape(ped_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
            dec_inp=batch['linear_traj'].float().to(device)
            out_input=batch['linear_traj'].float().to(device)

            src_mask = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
            spd_mask = torch.ones((inp.shape[0], 1, ego_patches)).to(device)
            ped_mask = torch.ones((inp.shape[0], 1, ped_patches)).to(device)

            out= model(out_input, inp,enc_extra_ego,enc_extra_ped,dec_inp,src_mask,spd_mask,ped_mask)
            out = out.to(device).cpu().numpy()
            gth = batch['pred_target'].to(device).cpu().numpy()

            gthtraj = np.add(np.multiply(gth, std_tr), mean_tr)
            predtraj = np.add(np.multiply(out, std_tr), mean_tr)

            gthtraj_center=np.concatenate((np.expand_dims((gthtraj[:,:,0]+gthtraj[:,:,2])/2,axis=-1),np.expand_dims((gthtraj[:,:,1]+gthtraj[:,:,3])/2,axis=-1)),axis=-1)
            predtraj_center = np.concatenate((np.expand_dims((predtraj[:, :, 0] + predtraj[:, :, 2]) / 2, axis=-1),np.expand_dims((predtraj[:, :, 1] + predtraj[:, :, 3]) / 2, axis=-1)), axis=-1)

            mse_batch_center_list.append(np.square(gthtraj_center-predtraj_center))
            mse_batch_list.append(np.square(gthtraj - predtraj))


        mse = np.concatenate(mse_batch_list, axis=0)
        mse_center=np.concatenate(mse_batch_center_list,axis=0)
        split_resultbytime(mse_center)
        print('MSE:',mse.mean(axis=None),'CMSE:',mse_center.mean(axis=None))
        return 0



if __name__=='__main__':
    model_path='JAAD_model/JAAD.pth'
    test(model_path)
