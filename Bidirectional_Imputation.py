import torch
from Imputation_Cell import Imputation_Cell

class Bidirectional_imputation(torch.nn.ModuleList):
    def __init__(self, input_size, hidden_size, layer_size, output_size, impute_len, dr_rate, device):
        super(Bidirectional_imputation, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.device = device
        self.output_size = output_size
        self.impute_len = impute_len
        self.dr_rate = dr_rate

        self.build()

    def build(self):
        self.lstm_forward = torch.nn.LSTM(self.input_size, self.hidden_size, self.layer_size, batch_first=True)
        self.lstm_backward = torch.nn.LSTM(self.input_size, self.hidden_size, self.layer_size, batch_first=True)
        self.impute_cell = Imputation_Cell(self.impute_len, self.input_size, self.hidden_size, self.dr_rate,
                                           self.output_size)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dr_rate),
            torch.nn.Linear(self.impute_len * 2, self.impute_len)
            # ,torch.nn.Tanh()
        )

    def forward(self, input_forward, input_backward, forward_perm_idx, backward_perm_idx, start_value, batchsize):

        batchsize = len(forward_perm_idx)

        h0_forward = torch.zeros(self.layer_size, batchsize, self.hidden_size).to(self.device)
        c0_forward = torch.zeros(self.layer_size, batchsize, self.hidden_size).to(self.device)
        h0_backward = torch.zeros(self.layer_size, batchsize, self.hidden_size).to(self.device)
        c0_backward = torch.zeros(self.layer_size, batchsize, self.hidden_size).to(self.device)

        _, forward_context = self.lstm_forward(input_forward, (h0_forward, c0_forward))
        _, backward_context = self.lstm_backward(input_backward, (h0_backward, c0_backward))
        forward_context = (forward_context[0].squeeze(0)[torch.argsort(forward_perm_idx)],
                           forward_context[1].squeeze(0)[torch.argsort(forward_perm_idx)])
        backward_context = (backward_context[0].squeeze(0)[torch.argsort(backward_perm_idx)],
                            backward_context[1].squeeze(0)[torch.argsort(backward_perm_idx)])

        impute_forward = self.impute_cell(batchsize, forward_context)
        impute_backward = torch.flip(self.impute_cell(batchsize, backward_context), [0])
        #         reverse_idx = torch.Tensor([idx for idx in reversed(range(impute_backward.shape[1]))]).long().to(self.device)
        #         print(impute_forward.shape)
        #         print(impute_backward.shape)

        final_impute = (impute_forward + impute_backward) / 2
        # final_impute = torch.clamp(self.fc_layer(torch.cat([impute_forward,impute_backward],dim=1)))

        return final_impute
