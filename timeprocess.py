from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# prepare data for lstm
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from pandas import DataFrame
import random
from sklearn.model_selection import train_test_split
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

def preprocessing(data, config):
    soft_border= config.soft_border
    train_size= config.train_size
    val_size= config.val_size
    test_size= config.test_size

    if config.scaling==True:
        scaler = StandardScaler()
        X = scaler.fit_transform(data)


    print('Creating Time sequence...')
    def create_sequences(data, config):
        seq_length, pred_length =  config.sequence_length, config.pred_len
        sequences = []
        targets = []
        for i in tqdm(range(len(data) - seq_length - pred_length + 1)):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length:i+seq_length+pred_length])
        return torch.tensor(sequences), torch.tensor(targets)

    X, y = create_sequences(X, config=config)

    if X.shape[2] > config.hidden_size:
        print("Here hidden size is: ", config.hidden_size, 'and data dimension is: ',  X.shape[2])
        raise ValueError("hidden_size should be equal or grater than features (dimension of data)")


    zeros = torch.ones((X.size(0), X.size(1),  config.hidden_size-X.shape[2]), dtype=X.dtype)

    # Concatenate along the last dimension
    X = torch.cat((X, zeros), dim=-1)

    batch = config.batch


    indices = np.arange(len(X))
    barrier = int(len(indices)/batch)*batch
    indices = indices[0:barrier]
    soft_border = int((config.sequence_length/batch))+soft_border

    indices = [indices[i:i+batch] for i in range(0, len(indices), batch)]

    border1 = int(len(indices)*train_size)
    border2 = border1+int(len(indices)*val_size)
    border3 = border2+int(len(indices)*test_size)

    train_ind = indices[0:border1]
    val_ind = indices[border1-soft_border: border2]
    test_ind = indices[border2-soft_border: border3]

    random.shuffle(train_ind)
    random.shuffle(val_ind)
    #random.shuffle(test_ind)


    X_train = [X[item] for sublist in train_ind for item in sublist]
    y_train = [y[item] for sublist in train_ind for item in sublist]

    X_val = [X[item] for sublist in val_ind for item in sublist]
    y_val = [y[item] for sublist in val_ind for item in sublist]

    X_test = [X[item] for sublist in test_ind for item in sublist]
    y_test = [y[item] for sublist in test_ind for item in sublist]

    print('Return as (X_train, y_train), (X_val, y_val), (X_test, y_test)')
  


    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class Dataset(Dataset):
    def __init__(self, tokenized_inputs,  labels=None, pos=None):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
        self.pos = pos
        self.id_list = None
        self.re = None

    def __len__(self):
        return len(self.tokenized_inputs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {
                "inputs_embeds": torch.tensor(self.tokenized_inputs[idx]),
                "labels_ids": torch.tensor(self.labels[idx]),
                #"id": torch.tensor(self.id_list[idx]),  # Include the id directly
                #"reservoir_ids": torch.tensor(self.re[idx]),
            }
        else:
            return {
                "inputs_embeds": torch.tensor(self.tokenized_inputs[idx]),
            }

class TimeTrainer(Trainer):
    def __init__(self, *args, gradient_accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = GradScaler()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss = loss / self.gradient_accumulation_steps
        self.scaler.scale(loss).backward()

        return loss.detach()


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset


        loader =  DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            shuffle = False,
        )
        return loader
