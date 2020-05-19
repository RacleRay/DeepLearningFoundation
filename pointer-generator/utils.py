# -*- coding:utf-8 -*-
# author: Racle
# project: pointer-network

import os
import pyrouge
import logging
import numpy as np
import torch
from torch.autograd import Variable
import tensorflow as tf

import config


def get_encoder_variables(batch, use_cuda):
    "创建encoder所需变量"
    batch_size = len(batch.enc_lens)

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    context_v_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

    coverage = None
    if config.do_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        context_v_1 = context_v_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, context_v_1, coverage


def get_decoder_variables(batch, use_cuda):
    "创建decoder所需的变量"
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        dec_padding_mask = dec_padding_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()
        target_batch = target_batch.cuda()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


def calc_moving_avg_loss(loss, moving_avg_loss, summary_writer, step, decay=0.99):
    "moving average loss"
    if moving_avg_loss == 0:  # on the first iteration just take the loss
        moving_avg_loss = loss
    else:
        moving_avg_loss = moving_avg_loss * decay + (1 - decay) * loss
    moving_avg_loss = min(moving_avg_loss, 12)  # clip

    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=moving_avg_loss)
    summary_writer.add_summary(loss_sum, step)

    return moving_avg_loss


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # doesn't correspond to an in-article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(log_str)


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    # print("Wrote example %i to file" % ex_index)


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s
