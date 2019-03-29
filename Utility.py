import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch.nn as nn


def train (model,dataloader,optim,loss_f,epoch,device,corr_value,norm_name=None):
    model.train()
    recon_dict = {'minmax': min_max_recover, 'zero': zero_norm_recover, 'total_zero': zero_norm_recover}
    mapping = {'minmax': 'min_max', 'zero': 'mean_var', 'total_zero': 'total_mean_var'}
    loss_list=[]
    raw_list=[]
    corr_list=[]
    output_list=[]

    for i, (corr,raw,mask,norm,stat_dict,demo_dict) in enumerate(dataloader):

        head_input,tail_input,label = input_split(norm,corr)
        head_padded_input = padding_input(head_input,720,)
        tail_padded_input =
        padded_label =

        if norm_name is not None:
            norm_corr = norm.clone()
            norm_corr[corr==-1]=corr_value
            encode = model.encoder(norm_corr)
            decode = model.decoder(encode)
            output_recon = recon_dict[norm_name](decode,stat_dict[mapping[norm_name]].to(device))
            #input_var = norm_corr.var(dim=1)

        else :
            encode = model.encoder(corr)
            decode = model.decoder(encode)
            output_recon = decode.copy()
            #input_var = corr.var(dim=1)

        origin = norm if norm_name is not None else raw
        output_var = decode.var(dim=1)
        #loss_var = torch.sqrt(torch.mean((output_var-input_var)**2))
        # 어떤 모델을 사용하느냐에 따라 모델의 로스 구성을 다르게 진행하여야 함

        # origin[corr!=-1]=0
        # decode[corr!=-1]=0

        optim.zero_grad()
        loss = loss_f(decode, origin)#+loss_var
        loss.backward()
        optim.step()
        loss_list.append(loss.cpu().item())

        raw_list.extend(raw.cpu().detach().numpy())
        corr_list.extend(corr.cpu().detach().numpy())
        output_list.extend(output_recon.cpu().detach().numpy())

    raw_list,corr_list,output_list = np.array(raw_list),np.array(corr_list),np.array(output_list)
    RMSE_total,RMSE_point = RMSE_F(raw_list,corr_list,output_list)
    MRE_total,MRE_point = MRE_F(raw_list,corr_list,output_list)
    MAE_total,MAE_point = MAE_F(raw_list,corr_list,output_list)

    print('{} epoch loss : {}'.format(epoch,np.array(loss_list).mean()))
    print('(train)RMSE:{:.2f} MAE:{:.2f} MRE:{:.2f}'.format(RMSE_point,MAE_point,MRE_point))





def test (model,dataloader,device,corr_value,norm_name=None):
    model.eval()
    recon_dict ={'minmax':min_max_recover,'zero':zero_norm_recover,'total_zero':zero_norm_recover}
    mapping = {'minmax':'min_max','zero':'mean_var','total_zero':'total_mean_var'}

    raw_list = []
    corr_list = []
    output_list = []

    for i, (corr, raw, mask, norm, stat_dict, demo_dict) in enumerate(dataloader):
        corr, raw, mask, norm = corr.to(device), raw.to(device), mask.to(device), norm.to(device)


        if norm_name is not None:

            norm_corr = norm.clone()
            norm_corr[corr==-1]=corr_value
            encode = model.encoder(norm_corr)
            output = model.decoder(encode)
            output = recon_dict[norm_name](output,stat_dict[mapping[norm_name]].to(device))
            #input_var = norm_corr.var(dim=1)
        else :
            encode = model.encoder(corr)
            output = model.decoder(encode)
            #input_var = corr.var(dim=1)

        raw_list.extend(raw.cpu().detach().numpy())
        corr_list.extend(corr.cpu().detach().numpy())
        output_list.extend(output.cpu().detach().numpy())

    raw_list,corr_list,output_list = np.array(raw_list),np.array(corr_list),np.array(output_list)
    RMSE_total,RMSE_point = RMSE_F(raw_list,corr_list,output_list)
    MRE_total,MRE_point = MRE_F(raw_list,corr_list,output_list)
    MAE_total,MAE_point = MAE_F(raw_list,corr_list,output_list)

    return {'RMSE':RMSE_total,'MRE':MRE_total,'MAE':MAE_total},{'RMSE':RMSE_point,'MRE':MRE_point,'MAE':MAE_point}


def visualizing(dataloader,model,device,norm_name,batch_size,save_path,corr_value,mode):
    recon_dict = {'minmax': min_max_recover, 'zero': zero_norm_recover, 'total_zero': zero_norm_recover}
    mapping = {'minmax': 'min_max', 'zero': 'mean_var', 'total_zero': 'total_mean_var'}

    for i, (corr, raw, mask, norm, stat_dict, demo_dict) in enumerate(dataloader):
        corr, raw, mask, norm = corr.to(device), raw.to(device), mask.to(device), norm.to(device)

        if norm_name is not None:

            norm_corr = norm.clone()
            norm_corr[corr==-1]=corr_value
            encode = model.encoder(norm_corr)
            output = model.decoder(encode)
            output = recon_dict[norm_name](output,stat_dict[mapping[norm_name]].to(device))
            #input_var = norm_corr.var(dim=1)
        else :
            encode = model.encoder(corr)
            output = model.decoder(encode)
            #input_var = corr.var(dim=1)
        break

    save_path = save_path+'\\figure\\'
    pathlib.Path(save_path).mkdir(exist_ok=True)

    output_np = output.cpu().detach().numpy()
    corr_np = corr.cpu().detach().numpy()
    raw_np = raw.cpu().detach().numpy()

    output_np[corr_np != -1] = 0
    raw_np[corr_np != -1] = 0

    for i in range(batch_size):
        plt.figure(figsize=(12, 6))
        plt.plot(output_np[i], 'r:')
        plt.plot(corr_np[i])
        plt.plot(raw_np[i], 'g')
        plt.savefig('{}{}_{}'.format(save_path,mode,i))
        plt.close()

        if i>30:
            break


def input_split(inputs,corr):
    head_input = [ inputs[i][:(row==-1).nonzero()[0]]   for i,row in enumerate(corr)]
    tail_input =  [torch.flip(inputs[i][(row==-1).nonzero()[-1]+1:],[0]) for i,row in enumerate(corr)]
    label = [ inputs[i][row==-1]   for i,row in enumerate(corr)]
    return head_input,tail_input,label


def padding_input(input_list, max_len, fill_value, device):
    dummy = torch.zeros([len(input_list), max_len]).to(device)
    dummy[dummy == 0] = fill_value

    for idx, input_row in enumerate(input_list):
        dummy[idx, :len(input_row)] = input_row
    return dummy


def info_writer(info,savepath):
    print(info)
    file = open('\n{}//{}'.format(savepath,'info.txt'),'a+')
    file.write(info)
    file.close()

def zero_norm_recover(data, mean_var):
    mean = mean_var[:, 0].view(-1, 1)
    std = torch.sqrt(mean_var[:, 1].view(-1, 1))


    recover = (data * std) + mean
    return recover


def min_max_recover(data, min_max):
    min = min_max[:, 0].view(-1, 1)
    max = min_max[:, 1].view(-1, 1)

    recover = data * (max - min) + min
    return recover

def RMSE_F(raw,corr,recover):
    mse_total = np.sqrt(np.square(raw - recover).mean())
    mse_point = np.sqrt(np.square(raw[corr==-1]-recover[corr==-1]).mean())
    return mse_total,mse_point

def MRE_F(raw,corr,recover):

    mre_total = (np.abs(raw - recover)/(raw+0.00000000000000000001)).mean()
    mre_point = (np.abs(raw[corr == -1] - recover[corr == -1])/(raw[corr==-1]+0.00000000000000000001)).mean()

    return mre_total,mre_point

def MAE_F(raw,corr,recover):
    mae_total = np.abs(raw - recover).mean()
    mae_point = np.abs(raw[corr == -1] - recover[corr == -1]).mean()
    return mae_total,mae_point


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) :
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data)

