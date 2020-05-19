# -*- coding:utf-8 -*-
# author: Racle
# project: pointer-network

import os
import time
import sys

import tensorflow as tf
import torch

import config
from data_loader import Vocab, Batcher
from model import Model
from utils import get_encoder_variables, get_decoder_variables, calc_moving_avg_loss

use_cuda = config.use_gpu and torch.cuda.is_available()


class Evaluate(object):
    """在eval data上计算损失。"""
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = tf.summary.FileWriter(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        "train_one_batch不进行back propagation"
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, context_v, coverage = \
            get_encoder_variables(batch, use_cuda)
        # dec_lens_var：一个batch的decoder目标序列长度
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_decoder_variables(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        d_hc = self.model.reduce_state(encoder_hidden)  # decoder初始h，c

        step_losses = []
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
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.do_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, step]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.data[0]

    def run_eval(self):
        moving_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            moving_avg_loss = calc_moving_avg_loss(loss, moving_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                    iter, print_interval, time.time() - start, moving_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()


if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()