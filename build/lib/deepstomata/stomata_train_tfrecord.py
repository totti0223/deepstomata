#!/usr/bin/env python
# -*- coding: utf-8 -*-
#https://github.com/WalkingMask/tMNIST
import sys
import tensorflow as tf
from datetime import datetime
import time
import os
from math import sqrt

import stomata_input
import stomata_model

LOGDIR = "/Users/todayousuke/Desktop/objectdetection/bm_icp_v2/bm_IcP/cnn/log"
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', '/Users/todayousuke/Desktop/objectdetection/bm_icp_v2/bm_IcP/cnn/stomata.tfrecords_ver1.1', 'File name of train data')
#flags.DEFINE_string('test', '/Users/kinoshitatoshinori-e01/Desktop/hogtrain/new2/3classtest.csv', 'File name of train data')
flags.DEFINE_string('train_dir', LOGDIR, 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 200, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

def main(ckpt = None):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder("float")
        images, labels = stomata_input.load_tf_data([FLAGS.train], FLAGS.batch_size, shuffle = True, distored = True)
        logits = stomata_model.tf_inference(images, FLAGS.batch_size, stomata_input.DST_INPUT_SIZE, stomata_input.NUM_CLASS)
        loss_value = stomata_model.tf_loss(logits, labels, FLAGS.batch_size, stomata_input.NUM_CLASS)
        train_op = stomata_model.tf_train(loss_value, global_step)
        #acc = stomata_model.accuracy(logits, labels)
        #images_test, labels_test, _ = stomata_input.load_data([FLAGS.train], FLAGS.batch_size, shuffle = True, distored = False)
        #logits_test = stomata_model.inference_deep(images, keep_prob, stomata_input.DST_INPUT_SIZE, stomata_input.NUM_CLASS)
        #acc_testacc_test = stomata_model.accuracy(logits_test, labels_test)
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(max_to_keep = 0)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if ckpt:
            #print 'restore ckpt', ckpt
            saver.restore(sess, ckpt)
        tf.train.start_queue_runners(sess=sess)
        #summary_op_train = tf.merge_summary([loss_op,acc_op_train,input_sum,h_conv1_sum,h_pool1_sum,image_op_train])
        #summary_op_test = tf.merge_summary([image_op_test,acc_op_test])

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_result = sess.run([train_op, loss_value], feed_dict={keep_prob: 0.6})
            duration = time.time() - start_time
            print (labels)
            exit()
            #acc_res_test = sess.run(acc_test, feed_dict={keep_prob: 1.0})

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_result, examples_per_sec, sec_per_batch))
                #print ('acc_res', acc_res)
                #print ('acc_test', acc_res_test)

            if step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict={keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)

                #summary_str = sess.run(summary_op_test,feed_dict={keep_prob: 1.0})
                #summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps or loss_result == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if loss_result == 0:
                print('loss is zero')
                break

if __name__ == '__main__':
    ckpt = None
    if len(sys.argv) == 2:
        ckpt = sys.argv[1]
    main()
    print ("done")

