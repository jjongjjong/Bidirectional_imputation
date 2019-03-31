import torch


class Imputation_Cell(torch.nn.ModuleList):
    def __init__(self, impute_len, input_size, hidden_size, dr_rate=0.3, output_size=1):
        super(Imputation_Cell, self).__init__()

        self.impute_len = impute_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dr_rate = dr_rate

        self.build()

    def build(self):
        self.LSTM_inpute = torch.nn.LSTMCell(self.input_size, self.hidden_size)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Dropout(self.dr_rate),
            torch.nn.Linear(self.hidden_size, self.input_size),
            torch.nn.Tanh(),
        )

    def forward(self, batch_size, context, start_value=0):
        output_seq = torch.empty(batch_size, self.impute_len)

        x = torch.zeros(batch_size, 1)
        for t in range(self.impute_len):
            h, c = self.LSTM_inpute(x.clone(), context)
            x = torch.clamp(self.fc_layer(h), min=0)
            context = (h, c)
            output_seq[:, t] = x.view(-1)
        return output_seq


