import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def train (model, dataloader, optim, loss_f, epoch, device, start_value, norm_name=None, complete_data_len=720, impute_max_len=50):
    model.train()
    recon_dict = {'minmax': min_max_recover, 'zero': zero_norm_recover, 'total_zero': zero_norm_recover}
    mapping = {'minmax': 'min_max', 'zero': 'mean_var', 'total_zero': 'total_mean_var'}
    loss_list=[]
    raw_list=[]
    corr_list=[]
    output_list=[]

    for i, (corr,raw,mask,norm,stat_dict,demo_dict) in enumerate(dataloader):

        input = norm if norm_name is not None else raw

        head_input, tail_input, label = input_split(input, corr)
        head_length_list = torch.Tensor([len(each) for each in head_input]).to(device)
        tail_length_list = torch.Tensor([len(each) for each in tail_input]).to(device)
        label_length_list = torch.Tensor([len(each) for each in label]).to(device)

        head_padded = padding_input(head_input, complete_data_len, 0, device)
        tail_padded = padding_input(tail_input, complete_data_len, 0, device)
        label_padded = padding_input(label, impute_max_len, -1, device)
        label_mask = torch.where(label_padded >= 0, label_padded, torch.zeros_like(label_padded))

        head_ordered_lengths, head_perm_idx = head_length_list.sort(0, descending=True)
        tail_ordered_lengths, tail_perm_idx = tail_length_list.sort(0, descending=True)

        head_padded = head_padded[head_perm_idx]
        tail_padded = tail_padded[tail_perm_idx]

        head_packed = pack_padded_sequence(head_padded.reshape(-1, complete_data_len, 1), head_ordered_lengths, batch_first=True)
        tail_packed = pack_padded_sequence(tail_padded.reshape(-1, complete_data_len, 1), tail_ordered_lengths, batch_first=True)

        output = model(head_packed, tail_packed, head_perm_idx, tail_perm_idx, start_value)
        output = output * label_mask

        optim.zero_grad()
        loss = loss_f(output, label_padded)#+loss_var
        loss.backward()
        optim.step()
        loss_list.append(loss.cpu().item())


        output_recon = recon_function(output, label_length_list, head_input, tail_input, complete_data_len)
        if norm_name is not None:
            output_recon = recon_dict[norm_name](torch.from_numpy(output_recon), stat_dict[mapping[norm_name]].to(device))

        raw_list.extend(raw.cpu().detach().numpy())
        corr_list.extend(corr.cpu().detach().numpy())
        output_list.extend(output_recon.numpy())

    raw_list,corr_list,output_recon = np.array(raw_list),np.array(corr_list),np.array(output_recon)
    RMSE_total,RMSE_point = RMSE_F(raw_list,corr_list,output_recon)
    MRE_total,MRE_point = MRE_F(raw_list,corr_list,output_recon)
    MAE_total,MAE_point = MAE_F(raw_list,corr_list,output_recon)

    print('{} epoch loss : {}'.format(epoch,np.array(loss_list).mean()))
    print('(train)RMSE:{:.2f} MAE:{:.2f} MRE:{:.2f}'.format(RMSE_point,MAE_point,MRE_point))




def test (model, dataloader, device, start_value, norm_name=None, complete_data_len=720, impute_max_len=50):
    model.eval()
    recon_dict ={'minmax':min_max_recover,'zero':zero_norm_recover,'total_zero':zero_norm_recover}
    mapping = {'minmax':'min_max','zero':'mean_var','total_zero':'total_mean_var'}

    raw_list = []
    corr_list = []
    output_list = []

    for i, (corr, raw, mask, norm, stat_dict, demo_dict) in enumerate(dataloader):

        input = norm if norm_name is not None else raw

        head_input, tail_input, label = input_split(input, corr)
        head_length_list = torch.Tensor([len(each) for each in head_input]).to(device)
        tail_length_list = torch.Tensor([len(each) for each in tail_input]).to(device)
        label_length_list = torch.Tensor([len(each) for each in label]).to(device)

        head_padded = padding_input(head_input, complete_data_len, 0, device)
        tail_padded = padding_input(tail_input, complete_data_len, 0, device)
        label_padded = padding_input(label, impute_max_len, -1, device)
        label_mask = torch.where(label_padded >= 0, label_padded, torch.zeros_like(label_padded))

        head_ordered_lengths, head_perm_idx = head_length_list.sort(0, descending=True)
        tail_ordered_lengths, tail_perm_idx = tail_length_list.sort(0, descending=True)

        head_padded = head_padded[head_perm_idx]
        tail_padded = tail_padded[tail_perm_idx]

        head_packed = pack_padded_sequence(head_padded.reshape(-1, complete_data_len, 1), head_ordered_lengths, batch_first=True)
        tail_packed = pack_padded_sequence(tail_padded.reshape(-1, complete_data_len, 1), tail_ordered_lengths, batch_first=True)

        output = model(head_packed, tail_packed, head_perm_idx, tail_perm_idx, start_value)
        output = output * label_mask

        output_recon = recon_function(output, label_length_list, head_input, tail_input, complete_data_len)
        if norm_name is not None:
            output_recon = recon_dict[norm_name](torch.from_numpy(output_recon), stat_dict[mapping[norm_name]].to(device))

    raw_list,corr_list,output_recon = np.array(raw_list),np.array(corr_list),np.array(output_recon)
    RMSE_total,RMSE_point = RMSE_F(raw_list,corr_list,output_recon)
    MRE_total,MRE_point = MRE_F(raw_list,corr_list,output_recon)
    MAE_total,MAE_point = MAE_F(raw_list,corr_list,output_recon)

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


def recon_function(output,label_len_list,head_input,tail_input,row_len):
    ''' imputation 부분과 head tail 부분을 합쳐서 리스트로 만들어 주는 함수
    :param output the np.array result of imputaion containing padding
    :param label_len_list
    :return the numpy array of fulfiled row data
    '''
    output = output.cpu().detach().numpy().tolist()
    label_len_list = label_len_list.cpu().numpy().tolist()

    recon_list =[]
    for i in range(len(output)):
        recon = head_input[i].numpy().tolist()+output[i][:int(label_len_list[i])]+tail_input+tail_input[i].numpy().tolist()
        assert len(recon)==row_len

        recon_list.append(recon)

    return np.array(recon_list)



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

