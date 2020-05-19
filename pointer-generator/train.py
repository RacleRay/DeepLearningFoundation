# -*- coding:utf-8 -*-
# author: Racle
# project: pointer-network

import os
import time
import argparse
import tqdm

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

import config
from data_loader import Vocab, Batcher
from utils import get_encoder_variables, get_decoder_variables, calc_moving_avg_loss


use_cuda = config.use_gpu and torch.cuda.is_available()


class Train:
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, moving_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': moving_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.do_coverage else config.lr
        self.optimizer = Adagrad(params,
                                 lr=initial_lr,
                                 initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            # 在训练到某个epoch，需要切换到coverage结构，因此需要使用新的optimizer状态。此处控制切换时机。
            if not config.do_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, context_v, coverage = \
            get_encoder_variables(batch, use_cuda)
        # dec_lens_var：一个batch的decoder目标序列长度
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_decoder_variables(batch, use_cuda)

        self.optimizer.zero_grad()

        if 0 in enc_lens:
            print('=================')
            print(enc_batch.shape)
            print(enc_lens)
            print(enc_batch)
            print('=================')
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        d_hc = self.model.reduce_state(encoder_hidden)  # decoder初始h，c

        step_losses = []
        # for step in tqdm.tqdm(range(min(max_dec_len, config.max_dec_steps))):
        for step in range(min(max_dec_len, config.max_dec_steps)):
            d_inp = dec_batch[:, step]  # Teacher forcing
            final_dist, d_hc, context_v, attn_dist, p_gen, next_coverage = self.model.decoder(d_inp,
                                                                                              d_hc,
                                                                                              encoder_outputs,
                                                                                              encoder_feature,
                                                                                              enc_padding_mask,
                                                                                              context_v,
                                                                                              extra_zeros,
                                                                                              enc_batch_extend_vocab,
                                                                                              coverage,
                                                                                              step)
            target = target_batch[:, step]
            # gather每一步target id的预测概率
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.do_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)  # encoder的累计分布作为损失，见原论文
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, step]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, moving_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        pbar = tqdm.tqdm(total = n_iters)
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            moving_avg_loss = calc_moving_avg_loss(loss, moving_avg_loss, self.summary_writer, iter)
            iter += 1
            pbar.update(1)

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(moving_avg_loss, iter)
        pbar.close()


if __name__ == '__main__':
    # model_file_path: 从头训练时设置为None，保存模型在log文件下的对应时间的train dir下。
    # 加载预训练的模型，在log文件下找。

    # parser = argparse.ArgumentParser(description="Train script")
    # parser.add_argument("-m",
    #                     dest="model_file_path",
    #                     required=False,
    #                     default=None,
    #                     help="Model file for retraining (default: None).")
    # args = parser.parse_args()

    train_processor = Train()
    # train_processor.trainIters(config.max_iterations, args.model_file_path)
    train_processor.trainIters(config.max_iterations)