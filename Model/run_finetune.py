# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append("..")
import optimization
import collections
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.metrics import roc_curve, auc, classification_report
from run_pretrain import *
import pandas as pd
import numpy as np
import pickle as pkl
import time

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example

def del_flags(FLAGS, keys_list):
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    return

def input_fn(input_files,
             is_training,
             num_cpu_threads=4):
    """ The actual input function"""

    name_to_features = {
        "address":
            tf.FixedLenFeature([1], tf.int64),
        "label":
            tf.FixedLenFeature([1], tf.float32),
        "input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_positions":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_counts":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_io_flags":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_values":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64)
    }

    if is_training:
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat(FLAGS.epoch).shuffle(100)

    else:
        d = tf.data.TFRecordDataset(input_files)

    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=FLAGS.batch_size)

    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()

    return features


def model_fn(features, mode, bert_config, vocab, init_checkpoint, learning_rate,
             num_train_steps, num_warmup_steps, load_cross, use_one_hot_embeddings):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("name = %s, shape = %s" % (name,
                                                   features[name].shape))

    label = tf.squeeze(features["label"]) # squeeze is important
    input_ids = features["input_ids"]
    input_positions = features["input_positions"]
    input_mask = features["input_mask"]
    input_io_flags = features["input_io_flags"]
    input_values = features["input_values"]
    input_counts = features["input_counts"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_positions=input_positions,
        input_io_flags=input_io_flags,
        input_amounts=input_values,
        input_counts=input_counts,
        input_mask=input_mask,
        token_type_ids=None,
        use_one_hot_embeddings=use_one_hot_embeddings,
        cross_share=FLAGS.cross_share)

    transformer_output = model.get_sequence_output()
    print(transformer_output)
    with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):

        # inp = tf.reduce_mean(transformer_output, 1)
        inp = transformer_output[:,0,:]

        dnn1 = tf.layers.dense(inp, FLAGS.hidden_size, activation=tf.nn.relu, name='f1')
        dnn2 = tf.layers.dense(dnn1, FLAGS.hidden_size, activation=tf.nn.relu, name='f2')
        logit = tf.squeeze(tf.layers.dense(dnn2 + dnn1, 1, activation=None, name='logit'))
        y_hat = tf.sigmoid(logit)

    # print("--------------------")
    # print("label:", label)
    # print("logit:", logit)
    # print("--------------------")
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    total_loss = loss
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if load_cross:
            # multi-layer parameter loading

            name_to_variable = collections.OrderedDict()
            for var in tvars:
                name = var.name
                m = re.match("^(.*):\\d+$", name)
                if m is not None:
                    name = m.group(1)
                name_to_variable[name] = var

            for layer_index in range(bert_config.num_hidden_layers):

                assignment_map = collections.OrderedDict()
                for name in name_to_variable.keys():
                    if "layer_" + str(layer_index) in name:

                        var_name_list = name.split("/")
                        var_name_list[2] = "shared_layer"
                        load_name = "/".join(var_name_list)
                        # assignment_map[name] = new_name
                        assignment_map[load_name] = name
                        initialized_variable_names[name] = 1
                        initialized_variable_names[name + ":0"] = 1

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, False)

        return model, train_op, total_loss

    elif mode == tf.estimator.ModeKeys.EVAL:

        return model, y_hat, total_loss

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def main(_):

    # load label
    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    def is_phish(address):
        if address in phisher_account_set:
            return 1.0
        else:
            return 0.0

    mode = tf.estimator.ModeKeys.TRAIN
    train_input_files = FLAGS.train_input_file
    train_features = input_fn(train_input_files, is_training=True)

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    # must have checkpoint
    if FLAGS.init_checkpoint==None:
        # raise ValueError("Must need a checkpoint for finetuning")
        print("No checkpoint!")

    train_bert_model, train_op, total_loss = model_fn(train_features, mode, bert_config, vocab,
                                                      FLAGS.init_checkpoint,
                                                      FLAGS.learning_rate,
                                                      FLAGS.num_train_steps, FLAGS.num_warmup_steps, FLAGS.load_cross, False)

    # saver define
    tvars = tf.trainable_variables()
    saver = tf.train.Saver(max_to_keep=30, var_list=tvars)

    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # start TRAINING
    losses = []
    iter = 0
    start = time.time()
    while True:
        try:
            _, loss = sess.run([train_op, total_loss])
            losses.append(loss)

            if iter % 100 == 0:
                end = time.time()
                loss = np.mean(losses)
                print("iter=%d, loss=%f, time=%.2fs" % (iter, loss, end - start))
                losses = []
                start = time.time()

            iter += 1

        except Exception as e:
            print("Out of Sequence")
            saver.save(sess, os.path.join(FLAGS.checkpointDir, "bert_finetune"))
            break

    # Evaluation
    mode = tf.estimator.ModeKeys.EVAL
    test_input_files = FLAGS.test_input_file
    test_features = input_fn(test_input_files, is_training=False)
    # do not load checkpoint
    test_bert_model, y_hat, total_loss = model_fn(test_features, mode, bert_config, vocab,
                                                  os.path.join(FLAGS.checkpointDir, "bert_finetune"),
                                                  FLAGS.learning_rate,
                                                  FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)

    address_id_list = []
    y_hat_list = []
    label_list = []

    iter = 0
    start = time.time()
    while True:
        try:
            address_id_v, y_hat_v, label_v, loss = sess.run([test_features["address"], y_hat, test_features["label"], total_loss])
            address_id_list += list(np.squeeze(address_id_v))
            y_hat_list += list(y_hat_v)
            label_list += list(label_v)
            losses.append(loss)

            if iter % 100 == 0:
                end = time.time()
                print("iter=%d, time=%.2fs" % (iter, end - start))
                start = time.time()

            iter += 1

        except Exception as e:
            print("Out of Sequence")
            # save model
            # saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(iter)))
            break

    sess.close()

    # generate final result
    address_id_list = np.array(address_id_list).reshape([-1])
    y_hat_list = np.array(y_hat_list).reshape([-1])
    label_list = np.array(label_list).reshape([-1])

    # aggregation
    # group by embedding according to address
    address_to_pred_proba = {}
    # address_to_label = {}
    for i in range(len(address_id_list)):
        address = address_id_list[i]
        pred_proba = y_hat_list[i]
        # label = label_list[i]
        try:
            address_to_pred_proba[address].append(pred_proba)
            # address_to_label[address].append(label)
        except:
            address_to_pred_proba[address] = [pred_proba]
            # address_to_label[address] = [label]

    # group to one
    address_list = []
    agg_y_hat_list = []
    agg_label_list = []

    for addr, pred_proba_list in address_to_pred_proba.items():
        address_list.append(addr)
        if len(pred_proba_list) > 1:
            agg_y_hat_list.append(np.mean(pred_proba_list, axis=0))
        else:
            agg_y_hat_list.append(pred_proba_list[0])

        agg_label_list.append(is_phish(vocab.id_to_tokens[addr]))

    # print("================ROC Curve====================")
    fpr, tpr, thresholds = roc_curve(agg_label_list, agg_y_hat_list, pos_label=1)
    print("AUC=", auc(fpr, tpr))

    print(np.sum(agg_label_list))
    print(np.sum(agg_y_hat_list))

    # for threshold in [0.01, 0.03, 0.05]:
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:

        print("threshold =", threshold)
        y_pred = np.zeros_like(agg_y_hat_list)
        y_pred[np.where(np.array(agg_y_hat_list) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(agg_label_list, y_pred, digits=4))

    return

if __name__ == '__main__':

    del_flags(FLAGS, ["do_train", "do_eval", "cross_share", "load_cross", "epoch", "max_seq_length", "train_input_file", "test_input_file", "init_checkpoint","learning_rate"])
    flags.DEFINE_bool("do_train", False, "")
    flags.DEFINE_bool("do_eval", True, "")
    flags.DEFINE_bool("cross_share", False, "whether to share or not")
    flags.DEFINE_bool("load_cross", True, "whether to load from cross")
    flags.DEFINE_integer("epoch", 1, "Epoch for finetune")
    flags.DEFINE_integer("max_seq_length", 100, "")
    flags.DEFINE_string("train_input_file", "./data/finetune_train.tfrecord", "Input train file for finetuning")
    flags.DEFINE_string("test_input_file", "./data/finetune_test.tfrecord", "Input test file for finetuning")
    flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
    flags.DEFINE_integer("hidden_size", 128, "Hidden size for downside MLP.")
    flags.DEFINE_float("learning_rate", 3e-4, "")

    print("==========Parameters===========")
    print("cross_share:", FLAGS.cross_share)
    print("load_cross:", FLAGS.load_cross)
    print("learning_rate:", FLAGS.learning_rate)
    tf.app.run()