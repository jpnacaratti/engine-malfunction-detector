#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:30:17 2024

@author: j.nacaratti
"""

import isodate
import os
import yt_dlp
import librosa
import soundfile as sf
import sounddevice as sd


def process_chunks(chunks, out_samplerate, yt_id, out_folder):
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
                saved_filename = f"CHUNK__{yt_id}__{count}__{i}.wav" 
            elif res == "w":
                saved_filename = f"CHUNK__W__{yt_id}__{count}__{i}.wav"
            elif res == "n" or res == "":
                print("Playing next CHUNK...")
                break
            elif res == "r":
                print("Repeating CHUNK...")
                continue
            else:
                continue
            
            out_path = os.path.join("dataset/noise_cutted", saved_filename)
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
            