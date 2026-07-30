# deepedmd_pytorch.py
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import _1_heter_helperfuns_LSTM as helper
from lstm_compact_dataset import get_train_val_test_from_compact
import matplotlib.pyplot as plt



# --------- Dataset & DataLoader ----------
def get_train_val_test(args, device):
    return get_train_val_test_from_compact(
        helper,
        args,
        args['batch_size'],
        device=device,
        path=args['data_path'],
    )

# --------- Model ----------
class Seq_to_Seq_LSTM(nn.Module):
    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device

        self.s_dim = args['s_dim']
        self.u_dim = args['u_dim']
        self.ep_dim = args['ep_dim']
        self.Npst = args['Npst']
        self.Nfut = args['Nfut']
        self.LSTM_dim = args['LSTM_dim'] # hidden/cell dimension

        # Encoder
        self.LSTM_encoder = nn.LSTM(input_size=self.s_dim, hidden_size=self.LSTM_dim, batch_first=True)
        
        # Decoder
        decoder_input_dim = self.s_dim + self.u_dim + self.ep_dim
        self.LSTM_decoder = nn.LSTM(input_size=decoder_input_dim, hidden_size=self.LSTM_dim, batch_first=True)
        # output projection
        self.fc = nn.Linear(self.LSTM_dim, self.s_dim)

        # move to device
        self.to(device)

    def encode(self, x_hist_seq: torch.Tensor, training: bool=False):
        """
        x_hist_seq: (batch_size, Npst, s_dim) 
        """
        # out:  (batch_size, Npst, LSTM_dim)
        # h_n:  (num_layers=1, batch_size, LSTM_dim)
        # c_n:  (num_layers=1, batch_size, LSTM_dim)
        out, (h_n, c_n) = self.LSTM_encoder(x_hist_seq)
        # Take the final step of the last layer
        # h_T: (batch_size, LSTM_dim)
        h_T = h_n[-1]

        # c_T: (batch_size, LSTM_dim)
        c_T = c_n[-1]

        return h_T, c_T

    def decode(self, h_T: torch.Tensor, c_T: torch.Tensor,
               x0: torch.Tensor, u_seq: torch.Tensor, ep_seq: torch.Tensor,
               training: bool=False):
        """
        h_T: (batch_size, LSTM_dim)   encoder hidden state
        c_T: (batch_size, LSTM_dim)   encoder cell state
        x0: (batch_size, s_dim)       current state
        u_seq: (batch_size, Nfut, u_dim)
        ep_seq: (batch_size, Nfut, ep_dim)
        """
        batch_size = h_T.size(0)

        # -------- Initial state --------
        # h_0: (1, batch_size, LSTM_dim)
        h = h_T.unsqueeze(0)

        # c_0: (1, batch_size, LSTM_dim)
        c = c_T.unsqueeze(0)

        outputs = []
        prev_x = x0
        
        for t in range(self.Nfut):
            # decoder_input: (batch_size, 1, s_dim + u_dim + ep_dim)
            decoder_input = torch.cat([prev_x, u_seq[:, t, :], ep_seq[:, t, :]], dim=-1).unsqueeze(1)

            # out: (batch_size, 1, LSTM_dim)
            # h:   (1, batch_size, LSTM_dim)
            # c:   (1, batch_size, LSTM_dim)
            out, (h, c) = self.LSTM_decoder(decoder_input, (h, c))

            # x_pred_t: (batch_size, s_dim)
            x_pred_t = self.fc(out).squeeze(1)

            outputs.append(x_pred_t)
            prev_x = x_pred_t

        # outputs: list of Nfut tensors, each (batch_size, s_dim)
        # After concatenation: (batch_size, Nfut, s_dim)
        return torch.stack(outputs, dim=1)

    def forward(self, x_hist_seq: torch.Tensor, u_seq: torch.Tensor, ep_seq: torch.Tensor, training: bool=False):
        h_T, c_T = self.encode(x_hist_seq, training=training)
        x0 = x_hist_seq[:, -1, :]
        return self.decode(h_T, c_T, x0, u_seq, ep_seq, training=training)
        
# --------- Loss computation ----------
def compute_losses(
    args: Dict,
    model: Seq_to_Seq_LSTM,
    x_seq: torch.Tensor,
    x_hist_seq: torch.Tensor,
    u_seq: torch.Tensor,
    ep_seq: torch.Tensor
):
    """
    x_hist_seq: (batch_size, Npst, s_dim)
    x_seq:      (batch_size, Nfut, s_dim)
    u_seq:      (batch_size, Nfut, u_dim)
    ep_seq:     (batch_size, Nfut, ep_dim)
    """

    # -------- Encoder --------
    # h_T, c_T: (batch_size, LSTM_dim)
    h_T, c_T = model.encode(x_hist_seq)
    x0 = x_hist_seq[:, -1, :]

    # -------- Decoder --------
    # x_pred: (batch_size, Nfut, s_dim)
    x_pred = model.decode(h_T, c_T, x0, u_seq, ep_seq)

    # -------- Loss (MSE) --------
    # element-wise MSE, then mean over all dims
    loss = torch.mean((x_pred - x_seq) ** 2)
    
    return loss, x_pred


# --------- training loop ----------
def training_loop(
    model: Seq_to_Seq_LSTM,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    args,
    device
):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_path = None
    epochs = args['epochs']
    patience = args['patience']
    wait = 0

    for epoch in range(epochs):
        # =======================
        # Train
        # =======================
        model.train()
        train_loss_epoch = 0.0

        for x_hist_seq, x_seq, u_seq, ep_seq in train_loader:
            # x_hist_seq: (batch, Npst, s_dim)
            # x_seq:      (batch, Nfut, s_dim)
            # u_seq:      (batch, Nfut, u_dim)
            # ep_seq:     (batch, Nfut, ep_dim)

            optimizer.zero_grad()

            loss, _ = compute_losses(
                args,
                model,
                x_seq=x_seq,
                x_hist_seq=x_hist_seq,
                u_seq=u_seq,
                ep_seq=ep_seq
            )

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * x_hist_seq.size(0)

        train_loss_epoch /= len(train_loader.dataset)
        train_losses.append(train_loss_epoch)

        # =======================
        # Validation
        # =======================
        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for x_hist_seq, x_seq, u_seq, ep_seq in val_loader:
                loss, _ = compute_losses(
                    args,
                    model,
                    x_seq=x_seq,
                    x_hist_seq=x_hist_seq,
                    u_seq=u_seq,
                    ep_seq=ep_seq
                )
                val_loss_epoch += loss.item() * x_hist_seq.size(0)

        val_loss_epoch /= len(val_loader.dataset)
        val_losses.append(val_loss_epoch)

        # =======================
        # Scheduler
        # =======================
        scheduler.step()

        # =======================
        # Early stopping
        # =======================
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            wait = 0
            best_model_path = args['model_repo_path']+f"/model_epoch_{epoch}.weights.h5"
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        # =======================
        # Logging
        # =======================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train LOSS: {train_loss_epoch:.6e} | "
                f"Val LOSS: {val_loss_epoch:.6e}"
            )

    return train_losses, val_losses, best_model_path


# --------- evaluation on test set ----------
def run_test_on_dataset(model: Seq_to_Seq_LSTM, test_loader, args, device):
    print("\n📊 Running evaluation on test set...")
    model.eval()
    with torch.no_grad():
        Nfut = args['Nfut']
        RMSE_vel = 0
        RMSE_spa = 0
        MAPE_vel = 0
        MAPE_spa = 0
        model.eval()

        for x_hist_seq, x_seq, u_seq, ep_seq in test_loader:

            loss, x_pred = compute_losses(
                    args,
                    model,
                    x_seq=x_seq,
                    x_hist_seq=x_hist_seq,
                    u_seq=u_seq,
                    ep_seq=ep_seq
                ) # x_pred: (batch_size, Nfut, s_dim)

            for i in range(Nfut):
                RMSE_vel += torch.sum((x_pred[:, i, 1::2] - x_seq[:, i, 1::2]) ** 2)
                RMSE_spa += torch.sum((x_pred[:, i, 0::2] - x_seq[:, i, 0::2]) ** 2)
                
                MAPE_vel += torch.sum(torch.abs(x_pred[:, i, 1::2] - x_seq[:, i, 1::2]) / (torch.abs(x_seq[:, i, 1::2])+1e-2))

                MAPE_spa += torch.sum(torch.abs(x_pred[:, i, 0::2] - x_seq[:, i, 0::2]) / (torch.abs(x_seq[:, i, 0::2])+1e-2))
        
        RMSE_vel = torch.sqrt(RMSE_vel / Nfut / len(test_loader.dataset))
        RMSE_spa = torch.sqrt(RMSE_spa / Nfut / len(test_loader.dataset)) 
        MAPE_vel = MAPE_vel / Nfut / len(test_loader.dataset) * 100.0
        MAPE_spa = MAPE_spa / Nfut / len(test_loader.dataset) * 100.0
        print(f"Sample count: {len(test_loader.dataset)}")   
        print(f"Test RMSE_vel: {RMSE_vel:.6f}, Test RMSE_spa: {RMSE_spa:.6f}, Test MAPE_vel: {MAPE_vel:.6f}%, Test MAPE_spa: {MAPE_spa:.6f}%\n")

    return RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa

# --------- helper to load trained model ----------
def load_trained_model(args, model_path, device):
    model = Seq_to_Seq_LSTM(args, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --------- predict plotting function ----------
def predict_plot(args, model, u_seq, ep_seq, x_seq_true, index, fig_path, device):
    """
    Convert to torch and create sliding windows
    
    Args:
        model: trained model
        u_seq: T * udim
        ep_seq: T * epdim
        x_seq_true: T+1 * xdim
        index: index for trajectory
        fig_path: path to save figure
        device: device to run on (cpu/gpu)
    """
    Nfut = args['Nfut']
    Npst = args['Npst']
    T = u_seq.shape[0]
    total_time_step = x_seq_true.shape[0]
    batch_size = total_time_step - Npst - Nfut + 1

    u_seq = torch.tensor(u_seq, dtype=torch.float32, device=device)
    ep_seq = torch.tensor(ep_seq, dtype=torch.float32, device=device)
    x_seq_true = torch.tensor(x_seq_true, dtype=torch.float32, device=device)

    u_batches, ep_batches, x_batches, x_hist_batches = [], [], [], []
    for i in range(batch_size):
        current_idx = i + Npst - 1
        u_slice = u_seq[current_idx:current_idx+Nfut] # Nfut * udim
        ep_slice = ep_seq[current_idx:current_idx+Nfut] # Nfut * epdim
        x_slice = x_seq_true[current_idx+1:current_idx+1+Nfut] # Nfut * x_dim
        x_hist_slice = x_seq_true[i:i+Npst] # Npst * x_dim

        u_batches.append(u_slice)
        ep_batches.append(ep_slice)
        x_batches.append(x_slice)
        x_hist_batches.append(x_hist_slice)

    u_seq_batch = torch.stack(u_batches, dim=0) # batch_size * Nfut * udim
    ep_seq_batch = torch.stack(ep_batches, dim=0)
    x_seq_true_batch = torch.stack(x_batches, dim=0)
    x_hist_seq_batch = torch.stack(x_hist_batches, dim=0)

    # print(f'u_seq_batch:{u_seq_batch.shape}, ep_seq_batch:{ep_seq_batch.shape}, x_seq_true_batch:{x_seq_true_batch.shape}, x_hist_seq_batch:{x_hist_seq_batch.shape}')

    y_pred = model(x_hist_seq_batch, u_seq_batch, ep_seq_batch, training=False)
    
    y_np = y_pred.detach().cpu().numpy()

    num_vehicle = args['num_CAV'] + args['num_HDV']

    fig, axs = plt.subplots(num_vehicle, 2, figsize=(10, 3*num_vehicle), sharex=True)
    fig.suptitle('Vehicle States Comparison - LSTM vs True', fontsize=12)

    for i in range(num_vehicle):
        axs[i,0].plot(np.arange(total_time_step), x_seq_true[:,2*i].cpu().numpy(), label='True', linewidth=1.2)
        for j in range(batch_size):
            axs[i,0].plot(Npst+np.arange(j, j+Nfut), y_np[j,:,2*i], 
            color='gray', linestyle='-', linewidth=0.8)
        axs[i,0].set_ylabel(f'Veh-{i} Spacing', fontsize=8)
        axs[i,0].tick_params(labelsize=8)
        axs[i,0].legend(fontsize=8)
        axs[i,0].grid(True, linestyle='--', linewidth=0.4, alpha=0.7)

        axs[i,1].plot(np.arange(total_time_step), x_seq_true[:,2*i+1].cpu().numpy(), label='True', linewidth=1.2)
        for j in range(batch_size):
            axs[i,1].plot(Npst+np.arange(j, j+Nfut), y_np[j,:,2*i+1], 
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
    return y_pred

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
    args = helper.get_args_local()
    args = vars(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args['is_train']:
        os.makedirs(args['path'], exist_ok=True)
        os.makedirs(args['model_repo_path'], exist_ok=True)
        helper.save_dict_as_txt(args, args['params_path'])

        train_loader, val_loader, test_loader = get_train_val_test(args, device)

        # check shapes
        for batch in train_loader:
            x_hist_seq, x_seq, u_seq, ep_seq = batch
            print("x_hist_seq shape:", x_hist_seq.shape)
            print("x_seq shape:", x_seq.shape)
            print("u_seq shape:", u_seq.shape)
            print("ep_seq shape:", ep_seq.shape)
            break

        model = Seq_to_Seq_LSTM(args, device=device)
        lr = args['learning_rate']
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=args['lr_step_size'], gamma=args['decay_rate'])

        train_losses, val_losses, best_model_path = training_loop(
            model, optimizer, scheduler, train_loader, val_loader, args, device
        )

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
        save_dir = os.path.dirname(os.path.dirname(model_path))  # parent of past_models/
        model = load_trained_model(args, model_path, device)

        # Compute test-set loss
        train_loader, val_loader, test_loader = get_train_val_test(args, device)
        RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa = run_test_on_dataset(model, test_loader, args, device)
        _save_test_metrics(helper, args, save_dir, model_path,
                           len(test_loader.dataset), RMSE_vel, RMSE_spa, MAPE_vel, MAPE_spa)
