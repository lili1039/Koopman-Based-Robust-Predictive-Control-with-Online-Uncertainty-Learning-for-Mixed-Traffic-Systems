# Run by:
# python _3_heter_mainfuns_highd_NoLSTM.py --Nfut 15 --Npst 50 --num_HDV 2 --is_train true

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import csv
import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.optim.lr_scheduler import StepLR

import _3_heter_helperfuns_highd_NoLSTM as helper
from highd_compact_dataset import get_train_val_test_from_compact
import matplotlib.pyplot as plt



# --------- Dataset & DataLoader ----------
class SeqDataset(Dataset):
    """
    Raw inputs:
        x_hist_data: (num_samples, T+1, Npst, slice_dim, num_HDV)
        x_data: (num_samples, T+1, s_dim)
        u_data: (num_samples, T, u_dim)
        ep_data:(num_samples, T, ep_dim)
    Return batches and transpose in collate to (T+1, B, ...)
    """
    def __init__(self, x_hist, x, u, ep):
        super().__init__()
        self.x_hist = x_hist.astype(np.float32)
        self.x = x.astype(np.float32)
        self.u = u.astype(np.float32)
        self.ep = ep.astype(np.float32)
        assert len(self.x_hist) == len(self.x) == len(self.u) == len(self.ep)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        """
        Return x_hist, x, u, and ep for sample idx.
        """
        return self.x_hist[idx], self.x[idx], self.u[idx], self.ep[idx]

def collate_fn(batch,device='cpu',dtype=torch.float32):
    """
    Input batch is a list.
    Each list element is x_hist(T+1,Npst,slice_dim,num_HDV), x(T+1,s_dim), u(T,u_dim), ep(T,ep_dim)
    """
    # batch: list of tuples
    x_hist = np.stack([b[0] for b in batch], axis=0)  # (B, T+1, Npst, slice_dim, num_HDV)
    x = np.stack([b[1] for b in batch], axis=0)       # (B, T+1, s_dim)
    u = np.stack([b[2] for b in batch], axis=0)       # (B, T, u_dim)
    ep = np.stack([b[3] for b in batch], axis=0)      # (B, T, ep_dim)

    # Transpose to the layout used by the TF code:(T+1, B, ...)
    x_hist = np.transpose(x_hist, (1,0,2,3,4))
    x = np.transpose(x, (1,0,2))
    u = np.transpose(u, (1,0,2))
    ep = np.transpose(ep, (1,0,2))

    # Convert to tensors and move to device.
    x_hist = torch.tensor(x_hist, dtype=dtype, device=device)
    x = torch.tensor(x, dtype=dtype, device=device)
    u = torch.tensor(u, dtype=dtype, device=device)
    ep = torch.tensor(ep, dtype=dtype, device=device)

    return x_hist, x, u, ep

def create_torch_dataloaders(x_hist_data, x_data, u_data, ep_data, batch_size, device='cpu'):
    """
        Receive data read from pkl and split into training, validation, and test sets;
        Return DataLoaders for the training, validation, and test sets
        x_hist_data: (num_samples, T+1, Npst, slice_dim, num_HDV)
        x_data: (num_samples, T+1, s_dim)
        u_data: (num_samples, T, u_dim)
        ep_data:(num_samples, T, ep_dim)
    """
    dataset = SeqDataset(x_hist_data, x_data, u_data, ep_data)
    # After DataLoader retrieves batch_size samples from the dataset,
    # it groups them into a list and passes it to collate_fn
    # Each list has length batch_size
    # Each list element is x_hist(T+1,Npst,slice_dim,num_HDV), x(T+1,s_dim), u(T,u_dim), ep(T,ep_dim)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
     # Use a lambda to pass device to collate_fn
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, device=device)
    )
    # The returned loader is iterable and can be used in training as: for x_hist, x, u, ep in loader:
    return loader

def get_train_val_test(batch_size, device='cpu', path=None, args=None):
    if args is None:
        args = vars(helper.get_args_local())
    return get_train_val_test_from_compact(helper, args, batch_size, device=device, path=path)

# --------- Model ----------
class DeepEDMD(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.s_dim = args['s_dim']
        self.u_dim = args['u_dim']
        self.ep_dim = args['ep_dim']
        self.lift_dim = args['lift_dim']
        self.Nfut = args['Nfut']
        self.num_HDV = args['num_HDV']

        # Encoder
        encoder_layers = []
        self.keep_prob = self.args['keep_prob']
        encoder_widths = self.args['encoder_widths']
        encoder_acts = self.args['eact_type']
        prev = encoder_widths[0]
        for width, act in zip(encoder_widths[1:], encoder_acts):
            encoder_layers.append(nn.Linear(prev, width))
            if act is not None and act.lower() in ['relu', 'tanh', 'sigmoid', 'leakyrelu']:
                if act.lower() == 'relu':
                    encoder_layers.append(nn.ReLU())
                elif act.lower() == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif act.lower() == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif act.lower() == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU())
            if self.keep_prob < 1.0:
                encoder_layers.append(nn.Dropout(p=1 - self.keep_prob))
            prev = width
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_widths = args['decoder_widths']
        w_in, w_out = decoder_widths[0], decoder_widths[1]
        self.decoder = nn.Linear(w_in, w_out, bias=False)

        # Koopman matrices
        total_state_dim = self.s_dim + self.lift_dim
        self.A = nn.Parameter(torch.randn(total_state_dim, total_state_dim) * math.sqrt(2/(total_state_dim + total_state_dim)))
        self.B = nn.Parameter(torch.randn(self.u_dim, total_state_dim) * math.sqrt(2/(self.u_dim + total_state_dim)))
        self.H = nn.Parameter(torch.randn(self.ep_dim, total_state_dim) * math.sqrt(2/(self.ep_dim + total_state_dim)))

        # move to device
        self.to(device)

    def encode(self, x: torch.Tensor, x_hist_seq: torch.Tensor, training: bool=False):
        """
        x: (batch, s_dim)
        x_hist_seq: (batch, Npst, HV_hist_slice_dim, num_HDV) 
        Run LSTM separately for each HDV: hist_i = x_hist_seq[:, :, :, i] -> (batch, Npst, HV_hist_slice_dim)
        """

        encoder_input = x  # (batch, s_dim)
        encoded = self.encoder(encoder_input)  # (batch, lift_dim)
        g = torch.cat([x, encoded], dim=-1)  # (batch, s_dim + lift_dim)

        return g

    def decode(self, g: torch.Tensor):
        out = self.decoder(g)
        return out

    def forward(self, x_seq, u_seq, ep_seq, x_history_seq, training: bool=False):
        """
        x_seq: (T+1, B, s_dim)
        u_seq: (T,   B, u_dim)
        ep_seq:(T,   B, ep_dim)
        x_history_seq: (T+1, B, Npst, slice_dim, num_HDV)
        returns: y_list (list of length Nfut+1, each (B, s_dim)),
                 g_list_pred (list of length Nfut+1, each (B, total_state_dim)),
                 g_list_true (list of length Nfut+1, each (B, total_state_dim))
        """
        x0 = x_seq[0]  # (B, s_dim)
        # x_history_seq[0] has shape (B, Npst, slice_dim, num_HDV)
        g0 = self.encode(x0, x_history_seq[0], training=training)  # (B, total_state_dim)
        g_list_pred = [g0]

        y_list = [self.decode(g0)]

        # compute true encodings for all time steps (Nfut+1)
        g_list_true = [self.encode(x_seq[t], x_history_seq[t], training=training) for t in range(self.Nfut + 1)]

        g = g0
        for t in range(self.Nfut):
            # shapes: g (B, total_state_dim), A (total_state_dim, total_state_dim)
            # u_seq[t]: (B, u_dim); u_seq[t] @ B -> (B, total_state_dim)
            g = g @ self.A + u_seq[t] @ self.B + ep_seq[t] @ self.H
            g_list_pred.append(g)
            y_list.append(self.decode(g))

        return y_list, g_list_pred, g_list_true

    def save_weights_to_csv(self, directory):
        os.makedirs(directory, exist_ok=True)

        np.savetxt(os.path.join(directory, "A.csv"), self.A.detach().cpu().numpy(), delimiter=",")
        np.savetxt(os.path.join(directory, "B.csv"), self.B.detach().cpu().numpy(), delimiter=",")
        np.savetxt(os.path.join(directory, "H.csv"), self.H.detach().cpu().numpy(), delimiter=",")

        # decoder first layer weight
        w = self.decoder.weight.detach().cpu().numpy()
        np.savetxt(os.path.join(directory, "decoder_weight.csv"), w, delimiter=",")

        print(f"✔️ Koopman weights and decoder weights saved to '{directory}/'")

# --------- Loss computation ----------
def compute_losses(args: Dict, model: DeepEDMD,
                   x_seq: torch.Tensor, u_seq: torch.Tensor, ep_seq: torch.Tensor,
                   y_list: List[torch.Tensor], g_list_true: List[torch.Tensor], g_list_pred: List[torch.Tensor]):
    """
    x_seq: (T+1, B, s_dim)
    y_list: list length T+1 of (B, s_dim)
    g_list_true/pred: list length T+1 of (B, total_state_dim)
    """

    Nfut = args['Nfut']
    state_weight = torch.tensor(args['state_weight'], device=x_seq.device)

    # ==================reconstruction loss======================
    loss_recon = torch.mean((y_list[0] - x_seq[0].squeeze(1))**2)
    for i in range(1, Nfut+1):
        temp_y = model.decode(g_list_true[i])
        loss_recon = loss_recon + torch.mean((temp_y - x_seq[i].squeeze(1))**2)
    loss_recon = args['recon_lam'] * loss_recon / (Nfut+1)

    # ==================prediction loss===========================
    loss_pred = 0.0
    RMSE_vel = 0
    RMSE_spa = 0

    for i in range(1, Nfut+1):
        loss_pred = loss_pred + torch.mean(((y_list[i] - x_seq[i].squeeze(1)) * state_weight) ** 2)
        RMSE_vel += torch.mean((y_list[i][:, 1::2] - x_seq[i,:,1::2]) ** 2)
        RMSE_spa += torch.mean((y_list[i][:, 0::2] - x_seq[i,:,0::2]) ** 2)
    loss_pred = args['predict_lam'] * loss_pred / Nfut
    RMSE_vel = torch.sqrt(RMSE_vel / Nfut)
    RMSE_spa = torch.sqrt(RMSE_spa / Nfut)

    # =====================linear loss============================
    loss_line = 0.0
    for i in range(1, Nfut+1):
        loss_line = loss_line + torch.mean((g_list_pred[i] - g_list_true[i])**2)
    loss_line = args['linear_lam'] * loss_line / Nfut

    # ===========Linf penalties==============================
    Linf1_penalty = torch.tensor(0.0, device=x_seq.device)
    # reconstruction loss (inf)
    for i in range(Nfut+1):
        if i == 0:
            Linf1_penalty = Linf1_penalty + torch.norm(torch.norm((y_list[0] - x_seq[0].squeeze(1)), dim=1, p=float('inf')), p=float('inf'))
        else:
            temp_y = model.decode(g_list_true[i])
            Linf1_penalty = Linf1_penalty + torch.norm(torch.norm((temp_y - x_seq[i].squeeze(1)), dim=1, p=float('inf')), p=float('inf'))

    Linf2_penalty = torch.tensor(0.0, device=x_seq.device)
    # prediction loss (inf)
    for i in range(1, Nfut+1):
        Linf2_penalty = Linf2_penalty + torch.norm(torch.norm(((y_list[i] - x_seq[i].squeeze(1)) * state_weight), dim=1, p=float('inf')), p=float('inf'))

    loss_Linf = args['Linf_lam'] * ((Linf1_penalty / (Nfut+1)) + (Linf2_penalty / Nfut))

    # =================L2 regularization on weights (exclude biases)==========================
    l2_loss = torch.tensor(0.0, device=x_seq.device)
    for name, p in model.named_parameters():
        if p.requires_grad and ('bias' not in name.lower()):
            l2_loss = l2_loss + torch.nn.functional.mse_loss(p, torch.zeros_like(p))
    loss_L2 = args['L2_lam'] * l2_loss

    total_loss = loss_recon + loss_pred + loss_line + loss_Linf + loss_L2

    loss_dict = {
        'recon': loss_recon,
        'pred': loss_pred,
        'line': loss_line,
        'linf': loss_Linf,
        'L2': loss_L2,
        'RMSE_vel': RMSE_vel,
        'RMSE_spa': RMSE_spa
    }
    return total_loss, loss_dict

# --------- training loop with torch.profiler integration ----------
def training_loop(model: DeepEDMD, optimizer, scheduler, train_loader, val_loader, args, device):
    best_val_loss = float('inf')
    best_model_path = None
    wait = 0
    epochs = args['epochs']
    patience = args['patience']

    train_losses = []
    val_losses = []
    os.makedirs(os.path.dirname(args['loss_path']), exist_ok=True)
    with open(args['loss_path'], 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'val_loss', 'train_recon', 'train_pred', 'train_line', 'train_linf', 'train_L2', 
        'train_RMSE_vel', 'train_RMSE_spa', 'val_pred', 'val_RMSE_vel', 'val_RMSE_spa']
        writer.writerow(header)

    # Note: create the profile object outside the loop and call prof.step() at each step
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")
        model.train()
        train_loss = 0.0
        train_count = 0
        train_loss_list = {'recon': 0.0, 'pred': 0.0, 'line': 0.0, 'linf': 0.0, 'L2': 0.0, 'RMSE_vel': 0.0, 'RMSE_spa': 0.0}

        # create a new profiler for each epoch so traces saved per-epoch
        # with profile(activities=activities, schedule=schedule, on_trace_ready=tensorboard_trace_handler(logdir),
        #              record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for batch_idx, (x_hist_seq, x_seq, u_seq, ep_seq) in enumerate(train_loader):

            optimizer.zero_grad()
            # with record_function("model_forward"):
            y_list, g_list_pred, g_list_true = model(x_seq, u_seq, ep_seq, x_hist_seq, training=True)
            loss_with_lam, loss_dict = compute_losses(args, model, x_seq, u_seq, ep_seq, y_list, g_list_true, g_list_pred)
            loss_with_lam.backward()
            optimizer.step()

            train_loss += loss_with_lam.item()
            for k in train_loss_list:
                train_loss_list[k] += loss_dict[k].item()
            train_count += 1

            if train_count % 5 == 0:
                avg = train_loss / train_count
                print(f"{train_count} batches have been learned. Current train loss = {loss_with_lam.item():.6f}. Avg so far = {avg:.6f}")

        # finalize epoch aggregated metrics
        if train_count > 0:
            train_loss /= train_count
            train_loss_list = {k: (train_loss_list[k] / train_count) for k in train_loss_list}
        else:
            train_loss = 0.0

        # validation
        model.eval()
        val_loss = 0.0
        val_loss_list = {'recon': 0.0, 'pred': 0.0, 'line': 0.0, 'linf': 0.0, 'L2': 0.0, 'RMSE_vel': 0.0, 'RMSE_spa': 0.0}
        val_count = 0
        with torch.no_grad():
            for x_hist_seq_val, x_seq_val, u_seq_val, ep_seq_val in val_loader:
                y_list, g_list_pred, g_list_true = model(x_seq_val, u_seq_val, ep_seq_val, x_hist_seq_val, training=False)
                loss_with_lam, loss_dict = compute_losses(args, model, x_seq_val, u_seq_val, ep_seq_val, y_list, g_list_true, g_list_pred)

                val_loss += loss_with_lam.item()
                for k in val_loss_list:
                    val_loss_list[k] += loss_dict[k].item()
                val_count += 1

        if val_count > 0:
            val_loss /= val_count
            val_loss_list = {k: (val_loss_list[k] / val_count) for k in val_loss_list}

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Train RMVE: {train_loss_list['RMSE_vel']:.6f}, Train RMSE: {train_loss_list['RMSE_spa']:.6f}")
        print(f"Val RMVE: {val_loss_list['RMSE_vel']:.6f}, Val RMSE: {val_loss_list['RMSE_spa']:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        with open(args['loss_path'], 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, train_loss, val_loss, train_loss_list['recon'], train_loss_list['pred'], train_loss_list['line'], train_loss_list['linf'], train_loss_list['L2'],
             train_loss_list['RMSE_vel'], train_loss_list['RMSE_spa'], val_loss_list['pred'], val_loss_list['RMSE_vel'], val_loss_list['RMSE_spa']]
            writer.writerow(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_path = args['model_repo_path']+f"/model_epoch_{epoch}.weights.h5"
            torch.save(model.state_dict(), best_model_path)
            print("✅ Saved best model")
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping")
                break

        scheduler.step()

    return train_losses, val_losses, best_model_path

# --------- evaluation on test set ----------
def run_test_on_dataset(model: DeepEDMD, test_loader, args, device):
    print("\n📊 Running evaluation on test set...")
    model.eval()
    with torch.no_grad():
        Nfut = args['Nfut']
        RMSE_vel = 0
        RMSE_spa = 0
        MAPE_vel = 0
        MAPE_spa = 0

        for x_hist_seq, x_seq, u_seq, ep_seq in test_loader:

            y_list, g_list_pred, g_list_true = model(x_seq, u_seq, ep_seq, x_hist_seq, training=False)

            for i in range(1, Nfut+1):
                RMSE_vel += torch.sum((y_list[i][:, 1::2] - x_seq[i,:,1::2]) ** 2)
                RMSE_spa += torch.sum((y_list[i][:, 0::2] - x_seq[i,:,0::2]) ** 2)
                

                MAPE_vel += torch.sum(torch.abs(y_list[i][:, 1::2] - x_seq[i,:,1::2]) / (torch.abs(x_seq[i,:,1::2])+1e-2))
                MAPE_spa += torch.sum(torch.abs(y_list[i][:, 0::2] - x_seq[i,:,0::2]) / (torch.abs(x_seq[i,:,0::2])+1e-2))

        RMSE_vel = torch.sqrt(RMSE_vel / Nfut / len(test_loader.dataset))
        RMSE_spa = torch.sqrt(RMSE_spa / Nfut / len(test_loader.dataset)) 
        MAPE_vel = MAPE_vel / Nfut / len(test_loader.dataset) * 100.0
        MAPE_spa = MAPE_spa / Nfut / len(test_loader.dataset) * 100.0
        print(f"Sample count: {len(test_loader.dataset)}")   
        print(f"Test RMSE_vel: {RMSE_vel:.6f}, Test RMSE_spa: {RMSE_spa:.6f}, Test MAPE_vel: {MAPE_vel:.6f}%, Test MAPE_spa: {MAPE_spa:.6f}%\n")

    return RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa

# --------- helper to load trained model ----------
def load_trained_model(args, model_path, device):
    model = DeepEDMD(args, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --------- predict plotting function ----------
def predict_plot(args, model, u_seq, ep_seq, x_seq_true, Time_step, model_path, weight_path, index, fig_path, device):
    """
    Convert to torch and create sliding windows
    
    Args:
        model: trained model
        u_seq: T * udim
        ep_seq: T * epdim
        x_seq_true: T+1 * xdim
        Time_step: time step information
        model_path: path to model
        weight_path: path to weights
        index: index for trajectory
        fig_path: path to save figure
        device: device to run on (cpu/gpu)
    """
    Nfut = args['Nfut']
    Npst = args['Npst']
    slice_dim = args['HV_hist_slice_dim']
    num_HDV = args['num_HDV']
    T = u_seq.shape[0]
    batch_size = T - Npst - Nfut + 1

    u_seq = torch.tensor(u_seq, dtype=torch.float32, device=device)
    ep_seq = torch.tensor(ep_seq, dtype=torch.float32, device=device)
    x_seq_true = torch.tensor(x_seq_true, dtype=torch.float32, device=device)

    u_batches, ep_batches, x_batches, x_hist_batches = [], [], [], []
    for i in range(batch_size):
        u_slice = u_seq[i+Npst:i+Npst+Nfut] # Nfut * udim
        ep_slice = ep_seq[i+Npst:i+Npst+Nfut] # Nfut * epdim
        x_slice = x_seq_true[i+Npst:i+Npst+Nfut+1] # Nfut+1 * x_dim

        x_hist_1dseq = torch.zeros((Nfut+1, Npst, x_slice.shape[1]))
        for j in range(Nfut+1):
            x_hist_1dseq[j] = x_seq_true[i+j:i+j+Npst]
        
        x_hist_slice = torch.zeros((Nfut+1, Npst, slice_dim, num_HDV), device=device, dtype=torch.float32) 
        # Nfut+1 * Npst * slice_dim * num_HDV
        for j in range(num_HDV):
            x_hist_slice[:,:,:,j] = x_hist_1dseq[:,:,(1+j*2):(slice_dim+1+j*2)]

        u_batches.append(u_slice)
        ep_batches.append(ep_slice)
        x_batches.append(x_slice)
        x_hist_batches.append(x_hist_slice)

    u_seq_batch = torch.stack(u_batches, dim=1) # Nfut * batch_size * udim
    ep_seq_batch = torch.stack(ep_batches, dim=1)
    x_seq_true_batch = torch.stack(x_batches, dim=1)
    x_hist_seq_batch = torch.stack(x_hist_batches, dim=1)

    # print(f'u_seq_batch:{u_seq_batch.shape}, ep_seq_batch:{ep_seq_batch.shape}, x_seq_true_batch:{x_seq_true_batch.shape}, x_hist_seq_batch:{x_hist_seq_batch.shape}')

    y_list, _, _ = model(x_seq_true_batch, u_seq_batch, ep_seq_batch, x_hist_seq_batch, training=False)
    
    # y_np = [y.detach().cpu().numpy() for y in y_list]  # list length Nfut+1 each (B, s_dim)
    # y_np = np.stack(y_list, axis=0)
    y_np = np.stack([y.detach().cpu().numpy() for y in y_list], axis=0)

    num_vehicle = args['num_CAV'] + args['num_HDV']
    total_time_step = T + 1

    fig, axs = plt.subplots(num_vehicle, 2, figsize=(10, 3*num_vehicle), sharex=True)
    fig.suptitle('Vehicle States Comparison - Koopman vs True', fontsize=12)

    for i in range(num_vehicle):
        axs[i,0].plot(np.arange(total_time_step), x_seq_true[:,2*i].cpu().numpy(), label='True', linewidth=1.2)
        for j in range(batch_size):
            axs[i,0].plot(Npst+np.arange(j, j+Nfut+1), y_np[:,j,2*i], 
            color='gray', linestyle='-', linewidth=0.8)
        axs[i,0].set_ylabel(f'Veh-{i} Spacing', fontsize=8)
        axs[i,0].tick_params(labelsize=8)
        axs[i,0].legend(fontsize=8)
        axs[i,0].grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

        axs[i,1].plot(np.arange(total_time_step), x_seq_true[:,2*i+1].cpu().numpy(), label='True', linewidth=1.2)
        for j in range(batch_size):
            axs[i,1].plot(Npst+np.arange(j, j+Nfut+1), y_np[:,j,2*i+1], 
            color='gray', linestyle='-', linewidth=0.8)
        axs[i,1].set_ylabel(f'Veh-{i} Velocity', fontsize=8)
        axs[i,1].tick_params(labelsize=8)
        axs[i,1].legend(fontsize=8)
        axs[i,1].grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

    axs[-1,0].set_xlabel('Time Step', fontsize=8)
    axs[-1,1].set_xlabel('Time Step', fontsize=8)
    # plt.tight_layout()
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, f'Traj_{index}.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure to {fig_path}")
    plt.close()
    return y_list

def _save_test_metrics(helper_mod, args, save_dir, best_model_path, num_samples,
                       RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa):
    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        'best_model_path': best_model_path,
        'num_samples': num_samples,
        'RMSE_vel': float(RMSE_vel),
        'RMSE_spa': float(RMSE_spa),
        'MAPE_vel': float(MAPE_vel),
        'MAPE_spa': float(MAPE_spa),
    }
    helper_mod.save_dict_as_txt(metrics, os.path.join(save_dir, 'test_metrics.txt'))


# --------- main ---------
if __name__ == '__main__':
    args = vars(helper.get_args_local())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args['is_train']:
        os.makedirs(args['path'], exist_ok=True)
        os.makedirs(args['model_repo_path'], exist_ok=True)
        batch_size = args['batch_size']

        helper.save_dict_as_txt(args, args['params_path'])

        train_loader, val_loader, test_loader = get_train_val_test_from_compact(
            helper, args, batch_size, device=device, path=args['data_path']
        )

        # check shapes
        for batch in train_loader:
            x_hist_seq, x_seq, u_seq, ep_seq = batch
            print("x_hist_seq shape:", x_hist_seq.shape)
            print("x_seq shape:", x_seq.shape)
            print("u_seq shape:", u_seq.shape)
            print("ep_seq shape:", ep_seq.shape)
            break

        model = DeepEDMD(args, device=device)
        lr = args['learning_rate']
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=args['lr_step_size'], gamma=args['decay_rate'])

        train_losses, val_losses, best_model_path = training_loop(
            model, optimizer, scheduler, train_loader, val_loader, args, device
        )

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.title('Training Loss Curve')
        plt.savefig(os.path.join(args['path'], 'loss_curve.png'))
        plt.close()

        # Auto-evaluate the best checkpoint on the test set and persist metrics.
        if best_model_path is not None:
            best_model = load_trained_model(args, best_model_path, device)
            RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa = run_test_on_dataset(best_model, test_loader, args, device)
            _save_test_metrics(helper, args, args['path'], best_model_path,
                               len(test_loader.dataset), RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa)
        else:
            print("⚠️ No best checkpoint was saved during training; skipping auto-eval.")
    else:
        model_path = args['eval_model_path']
        model_base_path = os.path.dirname(model_path) + os.sep
        weight_path = model_base_path + 'koopman_weights'
        save_dir = os.path.dirname(os.path.dirname(model_path))  # parent of past_models/

        model = load_trained_model(args, model_path, device)
        model.save_weights_to_csv(directory=weight_path)
        batch_size = args['batch_size']

        # Compute test-set loss
        train_loader, val_loader, test_loader = get_train_val_test_from_compact(
            helper, args, batch_size, device=device, path=args['data_path']
        )
        RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa = run_test_on_dataset(model, test_loader, args, device)
        _save_test_metrics(helper, args, save_dir, model_path,
                           len(test_loader.dataset), RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa)
