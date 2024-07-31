#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:36:58 2024

@author: j.nacaratti
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vggish')))

import tensorflow as tf
from vggish import vggish_input
from vggish import vggish_slim
from vggish import vggish_params

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()

def extract_embeddings(audio_path, hop_size=0.96):
    
    checkpoint_path = 'vggish/models/vggish_model.ckpt'
    
    vggish_slim.define_vggish_slim()
    vggish_params.EXAMPLE_HOP_SECONDS = hop_size
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

    input_batch = vggish_input.wavfile_to_examples(audio_path)

    return sess.run(embedding_tensor, feed_dict={features_tensor: input_batch})


embeddings = extract_embeddings('dataset/healthy_cutted/CHUNK__2PxDAswvYG4__0__0.wav', hop_size=(5 * 0.96))