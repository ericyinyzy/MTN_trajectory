import argparse
import baselineUtils
import torch
import torch.utils.data
import individual_TF
import os
import time
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
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
    parser.add_argument('--max_epoch',type=int, default=80)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--input_length', type=int, default=15)
    parser.add_argument('--ego_patches', type=int, default=64)
    parser.add_argument('--ped_patches', type=int, default=9)
    args=parser.parse_args()
    device=torch.device("cuda")
    if not os.path.exists('JAAD_model/{}'.format(datetime)):
        os.makedirs('JAAD_model/{}'.format(datetime))
    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")
    inp_l = args.input_length
    ego_patches = args.ego_patches
    ped_patches = args.ped_patches
    # creation of the dataloaders for train and validation
    mean_tr, std_tr = baselineUtils.generate_mean_std()
    train_dataset=baselineUtils.create_jaad_dataset(mean_tr,std_tr,dataset='jaad',flag='train',**args.model_opts)

    val_dataset=baselineUtils.create_jaad_dataset(mean_tr,std_tr,dataset='jaad',flag='val',**args.model_opts)


    model = individual_TF.IndividualTF(inp_l,4, 4, 4, N=args.layers,
                                       d_model=args.emb_size, d_ff=args.ff_size, h=args.heads, dropout=args.dropout,
                                       ).to(device)
    path='JAAD_model/warmup.pth'
    if not os.path.exists(path):
        raise ValueError("pretrained model does not exist")
    print("loading from pretrained model")
    pretrain=path
    with open(pretrain, "rb") as f:
        params = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=5, threshold=0, threshold_mode='rel', cooldown=0, min_lr=1e-07, eps=1e-08, verbose=True)
    epoch=0

    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()
        l=[]
        for id_b,batch in enumerate(tr_dl):
            optim.zero_grad()
            ego_op_flow=batch['ego_op_flow'].permute(0,1,3,2)
            enc_extra_ego=ego_op_flow[:,:(inp_l-1)].reshape(ego_op_flow.shape[0],2*(inp_l-1),-1).to(device)
            ped_op_flow = batch['ped_op_flow'].permute(0, 1, 3, 2)
            enc_extra_ped = ped_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
            inp = batch['enc_input'].float().to(device)
            out_input=batch['linear_traj'].float().to(device)
            dec_inp =batch['linear_traj'].float().to(device)

            src_mask = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            spd_mask=torch.ones((inp.shape[0],1, ego_patches)).to(device)
            ped_mask=torch.ones((inp.shape[0], 1, ped_patches)).to(device)


            # trg_att = torch.ones((inp.shape[0], 1,45)).to(device)
            pred=model(out_input,inp,enc_extra_ego,enc_extra_ped,dec_inp,src_mask,spd_mask,ped_mask)
            loss_fn = torch.nn.MSELoss()  # torch.nn.SmoothL1Loss()#torch.nn.MSELoss()
            loss= loss_fn(pred.float(), batch['pred_target'].float().to(device))
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            outPred = pred.to(device).cpu().detach().numpy()
            performance = np.square(outPred - batch['pred_target'].cpu().numpy())
            l.append(performance)

        mse_train = np.concatenate(l, axis=0)
        print("train epoch %03i/%03i epochloss: %7.4f "%(epoch, args.max_epoch,mse_train.mean(axis=None)))
        with torch.no_grad():
            model.eval()
            pr = []
            v=[]
            for id_b, batch in enumerate(val_dl):
                inp=batch['enc_input'].float().to(device)
                ego_op_flow = batch['ego_op_flow'].permute(0, 1, 3, 2)
                enc_extra_ego = ego_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
                ped_op_flow = batch['ped_op_flow'].permute(0, 1, 3, 2)
                enc_extra_ped = ped_op_flow[:, :(inp_l-1)].reshape(ego_op_flow.shape[0], 2*(inp_l-1), -1).to(device)
                out_input = batch['linear_traj'].float().to(device)
                dec_inp = batch['linear_traj'].float().to(device)
                # trg_att = torch.ones((inp.shape[0], 1, 45)).to(device)

                src_mask = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                spd_mask = torch.ones((inp.shape[0],1,ego_patches)).to(device)
                ped_mask = torch.ones((inp.shape[0], 1, ped_patches)).to(device)

                out = model(out_input,inp,enc_extra_ego,enc_extra_ped,dec_inp,src_mask,spd_mask,ped_mask)

                val_loss=out.float()-batch['pred_target'].float().to(device)
                val=val_loss.pow(2)
                out=out.to(device).cpu().numpy()
                performance = np.square(out - batch['pred_target'].cpu().numpy())
                pr.append(performance)
                v.append(val)
        mse_valid = np.concatenate(pr, axis=0)
        mse_valid=mse_valid.mean(axis=None)
        val_loss=torch.mean(torch.cat(v,axis=0))
        scheduler.step(val_loss)
        print('mse loss:{}'.format(mse_valid))
        if epoch+1==args.max_epoch:
            torch.save(model.state_dict(), f'JAAD_model/{datetime}/JAAD.pth')

        if epoch%10==0:
            torch.save(model.state_dict(), f'JAAD_model/{datetime}/{str(epoch)}.pth')


        epoch+=1



if __name__=='__main__':
    print('start_training')
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time_local)
    print(f'models are saved in JAAD_model/{datetime}')
    main()
