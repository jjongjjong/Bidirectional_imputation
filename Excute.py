from Activity_Dataset import activityDataset
from  torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from Bidirectional_Imputation import Bidirectional_imputation
from Utility import train,test,visualizing,info_writer
import datetime
import pathlib


input_size = 1
hidden_size = 80
layer_size = 1
output_size = 1
impute_len = 50
dr_rate = 0.2
corr_value=0.5
batch_size = 128

lr = 0.0003
epochs = 500
norm_name = 'minmax' #minmax/zero/total_zero/None
folder='30zero'


info = 'model_{} hidden_{} norm_{} corr_{} dr_{}'.format('LSTM_Input',hidden_size,norm_name,corr_value,dr_rate)
time = datetime.datetime.now().strftime('%Y%m%d-%H%M-%S')
save_path = 'E:\\Jupyter_notebook\\JJH\\dataStorage\\NHNES\\model\\{}_{}'.format(info,time)
#save_path = 'E:\\Jupyter_notebook\\JJH\\dataStorage\\NHNES\\model'
pathlib.Path(save_path).mkdir(exist_ok=True)
info_writer(info,save_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


root_dir = 'E:\\Jupyter_notebook\\JJH\\dataStorage\\NHNES\\preprocess\\'

tr_dataset = activityDataset(root_dir,folder,'train',normalize=norm_name)
vd_dataset = activityDataset(root_dir,folder,'valid',normalize=norm_name)
te_dataset = activityDataset(root_dir,folder,'test',normalize=norm_name)
print('dataset prepared')

tr_dataloader = DataLoader(tr_dataset,batch_size,shuffle=True,drop_last=True)
vd_dataloader = DataLoader(vd_dataset,batch_size,shuffle=False,drop_last=False)
te_dataloader = DataLoader(te_dataset,batch_size,shuffle=False,drop_last=False)
print('dataloader prepared')

#model = AutoEncoder(input_size,output_size).double().to(device)
model = Bidirectional_imputation().double().to(device)
# model.apply(weights_init)
# model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr = lr)
loss_f = torch.nn.MSELoss()

best_MSE_score = 1000000000000000
best_MAE_score = 1000000000000000
best_model_path = None

for epoch in range(epochs):
    train(model,tr_dataloader,optimizer,loss_f,epoch,device,corr_value,norm_name)
    total_dict,point_dict = test(model, vd_dataloader, device,corr_value,norm_name)
    print('RMSE:{:.2f}  MAE:{:.2f}  MRE:{:.2f}'.format(point_dict['RMSE'],point_dict['MAE'],point_dict['MRE']))
    if epoch>20 and (point_dict['RMSE']<best_MSE_score or point_dict['MAE']< best_MAE_score):
        good_model_path = "{}\\({})_{:.2f}_{:.2f}_{:.2f}.pt".format(save_path,epoch,point_dict['RMSE'],point_dict['MAE'],point_dict['MRE'])
        torch.save(model.state_dict(),good_model_path)

        if point_dict['RMSE']<best_MSE_score:
            best_MSE_score = point_dict['RMSE']
        elif point_dict['MAE'] < best_MAE_score:
            best_MAE_score = point_dict['MAE']

        best_model_path=good_model_path



model.load_state_dict(torch.load(best_model_path))
visualizing(tr_dataloader,model,device,norm_name,batch_size,save_path,corr_value,mode='train')
visualizing(vd_dataloader,model,device,norm_name,batch_size,save_path,corr_value,mode='valid')

test_total,test_point = test(model,te_dataloader,device,norm_name)
visualizing(te_dataloader,model,device,norm_name,batch_size,save_path,corr_value,mode='test')
info_writer(test_point,save_path)
print(test_point)
print('DONE!')
