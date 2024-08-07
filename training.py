import geoopt
from configs import parser
import json
import logging
import time
import os
import numpy as np
import torch


from models.autoencoder import Autoencoder
from optim import RiemannianAdam, select_optimizer
from utils.data_utils import load_data, load_batch
from utils.logging_utils import Logger

'''
TODO last: add things to config (encoder, decoder, loss function, path, hyp_lr and hyp_weight_decay, etc)
'''

class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        ### TODO second to last: add loader for MNIST dataset
        self.train_loader, self.test_loader = load_data(self.args)
    def train_AE(self):
        model = Autoencoder(args)
        loss_function = torch.nn.MSELoss()
        optimizer, lr_scheduler = select_optimizer.select(args, model)
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        print("total number of autoencoder parameters: " + str(tot_params))
        model = model.to(args.device)
        # for early stopping, can be added later
        # counter = 0 
        best_mean_test_loss = 1e10
        logger = Logger(str(os.path.join(self.args.save_dir, f'{self.args.checkpoint_path}.log')), mode='a')
        for epoch in range(args.epochs):
            t_start = time.time()
            model.train()
            self.total_train_loss = []
            self.total_test_loss = []

            for batch, step in self.train_loader:
                optimizer.zero_grad()
                batch = load_batch(self.args, batch)
                emb = model(batch)
                if self.args.encoder_loss:
                    loss = loss_function(emb, batch)
                else:
                    loss = loss_function(emb[1], batch)
                loss.backward()

                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                self.total_train_loss.append(loss.item())

            with torch.no_grad():
                model.eval()
                for test_batch, _ in self.test_loader:
                    test_batch = load_batch(self.args, test_batch)
                    emb = model(test_batch)
                    if self.args.encoder_loss:
                        loss = loss_function(emb, test_batch)
                    else:
                        loss = loss_function(emb[1], test_batch)
                    self.total_test_loss.append(loss.item())
            mean_total_train_loss = np.mean(self.total_train_loss)
            mean_total_test_loss = np.mean(self.total_test_loss)
            if (epoch + 1) % self.args.train_save_freq == 0 and best_mean_test_loss>mean_total_test_loss:
                best_mean_test_loss = mean_total_test_loss 
                torch.save({
                    'epoch': epoch,
                    'AE_state': model.state_dict()
                }, f'./checkpoints/{self.args.checkpoint_path}/{epoch}.pth' )

            if (epoch + 1) % self.args.log_freq == 0:
                logger.log(f'{epoch + 1:03d} | {time.time() - t_start:.2f}s | '
                           f'total train loss: {mean_total_train_loss:.3e} | '
                           f'total test loss: {mean_total_test_loss:.3e} ', verbose=False)
        print(' ')

    def train_sde(self):
        



if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    trainer = Trainer(args)
    trainer.train_AE()