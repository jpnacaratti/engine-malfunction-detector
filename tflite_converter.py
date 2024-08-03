#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 22:17:50 2024

@author: j.nacaratti
"""

import tensorflow as tf
from utils import load_vggish_model


###################################
# Converting classifier to tflite #
###################################

savedmodel_path = 'models/classifier__bs_80__epoch_115__val_0.8783'
classifier_tflite_save_path = 'models/fault_classifier.tflite'

classifier_converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
classifier_tflite_model = classifier_converter.convert()

with open(classifier_tflite_save_path, 'wb') as f:
    f.write(classifier_tflite_model)

print("Fault classifier successfully converted to tflite format in: {classifier_tflite_save_path}")


###############################
# Converting VGGish to tflite #
###############################

hop_size = 0.96 # seconds
vggish_checkpoint_path = 'vggish/models/vggish_model.ckpt'
vggish_savedmodel_dir = 'vggish/models/vggish_savedmodel'
vggish_tflite_save_path = 'vggish/models/vggish.tflite'

sess, input_tensor, output_tensor = load_vggish_model(hop_size, vggish_checkpoint_path)

builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(vggish_savedmodel_dir)

signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': input_tensor},
        outputs={'output': output_tensor})

builder.add_meta_graph_and_variables(
        sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={'serving_default': signature})

builder.save()

print(f"VGGish saved as Saved Model in: {vggish_savedmodel_dir}")

vggish_converter = tf.lite.TFLiteConverter.from_saved_model(vggish_savedmodel_dir)
vggish_tflite_model = vggish_converter.convert()

with open(vggish_tflite_save_path, 'wb') as f:
    f.write(vggish_tflite_model)

print(f"VGGish successfully converted to tflite format in: {vggish_tflite_save_path}")