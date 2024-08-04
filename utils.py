#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:30:17 2024

@author: j.nacaratti
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vggish')))

import isodate
import yt_dlp
import librosa
import threading
import json
import soundfile as sf
import sounddevice as sd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from vggish import vggish_input
from vggish import vggish_slim
from vggish import vggish_params

def convert_training_info_to_dict(accuracy, val_accuracy, val_loss, model_path, combo):
    neurons, kernel_init, kernel_reg, dropout, third_block, fourth_block, epochs, batch_size = combo
    
    result = {
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'model_path': model_path,
        'combo': {
            'neurons': neurons,
            'kernel_init': kernel_init,
            'kernel_reg': kernel_reg,
            'dropout': dropout,
            'third_block': third_block,
            'fourth': fourth_block,
            'epochs': epochs,
            'batch_size': batch_size
        }
    }
    
    return result

def dict_to_json_file(to_save, save_path):
    try:
        with open(save_path, 'w') as json_file:
            json.dump(to_save, json_file, indent=4)
    except Exception as e:
        print(f"Error saving the dict in {save_path}: {e}")

def classify_tflite(interpreter, input_details, output_details, embeddings):
    interpreter.set_tensor(input_details[0]['index'], embeddings)
    
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

def extract_embeddings_tflite(interpreter, input_details, output_details, audio_path):
    input_batch = vggish_input.wavfile_to_examples(audio_path)
    
    input_batch = input_batch.astype(np.float32)
    
    embeddings = []
    for spectogram in input_batch:
        spectogram = np.expand_dims(spectogram, axis = 0)
    
        interpreter.set_tensor(input_details[0]['index'], spectogram)
    
        interpreter.invoke()
        
        embedding = interpreter.get_tensor(output_details[0]['index'])
        embeddings.append(embedding)
    
    return np.vstack(embeddings)

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

def plot_training_history(history, save_path = None):
    plt.figure(figsize = (12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss during training')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'], label = 'Training accuracy')
    plt.plot(history.history['val_binary_accuracy'], label = 'Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy during training')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def extract_all_features_from_folder(folder_path, has_noise, hop_size, checkpoint_path, embeddings_type = 'all'):
    rows = []

    for element in tqdm(os.listdir(folder_path)):
        
        audio_path = os.path.join(folder_path, element)
        
        embeddings = extract_embeddings(audio_path, hop_size, checkpoint_path)
        
        if embeddings_type == 'mean':
            rows.append([audio_path, int(has_noise)] + np.mean(embeddings, axis = 0).tolist())
        elif embeddings_type == 'first':
            rows.append([audio_path, int(has_noise)] + embeddings[0].tolist())
        elif embeddings_type == 'all':
            for features in embeddings:
                rows.append([audio_path, int(has_noise)] + features.tolist())
    
    return rows

def load_vggish_model(hop_size, checkpoint_path):
    
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    
    vggish_slim.define_vggish_slim()
    vggish_params.EXAMPLE_HOP_SECONDS = hop_size
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    
    return sess, features_tensor, embedding_tensor

def extract_embeddings(audio_path, hop_size, checkpoint_path):
    
    sess, features_tensor, embedding_tensor = load_vggish_model(hop_size, checkpoint_path)

    input_batch = vggish_input.wavfile_to_examples(audio_path)

    return sess.run(embedding_tensor, feed_dict={features_tensor: input_batch})

def process_chunks(chunks, out_samplerate, yt_id, out_folder, prefix = 'CHUNK__'):
    count = 0
    for i in range(len(chunks)):
        chunk = chunks[i]
        
        while True:
            
            player = AsyncAudioPlayer()
            player.play(chunk, out_samplerate)
            
            res = input(f"Analising CHUNK = {i}. Proceed with the audio? y|w|n|r ")
            
            player.stop()
            
            if res == "y":
                # CHUNK_{YT_VIDEO_ID}_{VIDEO_ID_FILES_SAVED}_{CHUNK_SAVED}
                saved_filename = f"{prefix}{yt_id}__{count}__{i}.wav" 
            elif res == "w":
                saved_filename = f"{prefix}W__{yt_id}__{count}__{i}.wav"
            elif res == "n" or res == "":
                print("Playing next CHUNK...")
                break
            elif res == "r":
                print("Repeating CHUNK...")
                continue
            else:
                continue
            
            out_path = os.path.join(out_folder, saved_filename)
            sf.write(out_path, chunk, out_samplerate)
            count += 1
            
            print(f"CHUNK saved as: {saved_filename}")
            break

def cut_audio(audio_path, seconds):
    y, sr = librosa.load(audio_path, sr=None)
    
    duration = librosa.get_duration(y=y, sr=sr)
    
    chunks = int(duration) // seconds
    
    to_return = []
    for i in range(chunks):
        start_position = i * sr * seconds
        audio_length = seconds * sr
        chunk = y[start_position:start_position + audio_length]
        
        to_return.append(chunk)
        
    return to_return

def resample_audio(in_path, out_path, samplerate):
    y, sr = librosa.load(in_path, sr=None)
    
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=samplerate)
    
    sf.write(out_path, y_resampled, samplerate)

def download_youtube_audio(video_id, dest_folder):
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    
    file_path = os.path.join(dest_folder, f'{video_id}.%(ext)s')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': file_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }]
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def fetch_youtube_ids(queries, max_results, yt_service):
    
    ids = []
    titles = []
    durations = []
    
    for query in queries:
        request = yt_service.search().list(
            part="id,snippet",
            type='video',
            q=query,
            videoDuration='short',
            order="relevance",
            maxResults=10,
            fields="items(id(videoId),snippet(title))"
        )
        
        # {'items': [{'id': {'videoId': 'XXXXXXXX'}, 'snippet': {'title': 'Video title here'}}]}
        response = request.execute()
        print(response)
        
        extracted_ids = list(map(lambda kv: kv['id']['videoId'], response['items']))
        extracted_titles = list(map(lambda kv: kv['snippet']['title'], response['items']))
        extracted_durations = []
        
        for yt_id in extracted_ids:
            d_request = yt_service.videos().list(
                part="contentDetails",
                id=yt_id,
                fields="items(contentDetails(duration))"
            )
            
            # {'items': [{'contentDetails': {'duration': 'PT3M23S'}}]}
            d_response = d_request.execute()
            print(d_response)
            
            d_iso = d_response['items'][0]['contentDetails']['duration']
            d_seconds = int(isodate.parse_duration(d_iso).total_seconds())
            
            extracted_durations.append(d_seconds)
        
        
        ids.extend(extracted_ids)
        titles.extend(extracted_titles)
        durations.extend(extracted_durations)
        
    return ids, titles, durations

class AsyncAudioPlayer:
    def __init__(self):
        self._thread = None
        self._running = False
        
    def _play_audio(self, chunk, samplerate):
        sd.play(chunk, samplerate)
        
    def play(self, samples, samplerate):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self._play_audio, args=(samples, samplerate))
            self._thread.start()
            
    def stop(self):
        if self._running:
            self._running = False
            sd.stop()
            self._thread.join()
            