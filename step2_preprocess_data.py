"""
Step 2: Data Preprocessing Pipeline
====================================
This script processes videos from AV-Deepfake1M (or any video dataset).
For each video it:
  1. Extracts face crops using MTCNN (aligned, 224x224 for CLIP)
  2. Extracts audio waveform (resampled to 16kHz for WavLM)
  3. Saves everything in a structured format

BEFORE RUNNING:
  - Download some videos from AV-Deepfake1M into data/raw/
  - Or use ANY video files to test the pipeline first
  
  Dataset structure expected:
    data/raw/
      real/
        video001.mp4
        video002.mp4
      fake/
        video003.mp4
        video004.mp4

  Output structure created:
    data/processed/
      real/
        video001/
          faces/frame_0000.jpg, frame_0001.jpg, ...
          audio.wav
          metadata.json
      fake/
        video003/
          faces/frame_0000.jpg, frame_0001.jpg, ...
          audio.wav
          metadata.json

Usage: python step2_preprocess_data.py --input data/raw --output data/processed --max_frames 16
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import torchaudio
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN


def setup_face_detector(device='cuda'):
    """Initialise MTCNN face detector"""
    mtcnn = MTCNN(
        image_size=224,       # CLIP expects 224x224
        margin=40,            # Extra margin around face (captures more context)
        min_face_size=50,     # Minimum face size to detect
        thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
        factor=0.709,
        post_process=False,   # Don't normalise — we'll do it ourselves
        device=device
    )
    return mtcnn


def extract_faces(video_path, output_dir, mtcnn, max_frames=16):
    """
    Extract aligned face crops from a video.
    
    Args:
        video_path: path to video file
        output_dir: directory to save face crops
        mtcnn: MTCNN face detector
        max_frames: number of frames to sample (evenly spaced)
    
    Returns:
        dict with metadata about extracted faces
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠️  Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames < max_frames:
        # If video has fewer frames than requested, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    extracted = []
    failed = 0
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            failed += 1
            continue
        
        # Convert BGR to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect and align face
        try:
            face = mtcnn(pil_image)
            if face is not None:
                # face is a tensor of shape (3, 224, 224) with values 0-255
                # Save as image
                face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)
                face_pil = Image.fromarray(face_np)
                save_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
                face_pil.save(save_path, quality=95)
                extracted.append({
                    'frame_idx': frame_idx,
                    'sample_idx': idx,
                    'timestamp': frame_idx / fps if fps > 0 else 0,
                    'path': f"frame_{idx:04d}.jpg"
                })
            else:
                failed += 1
        except Exception as e:
            failed += 1
    
    cap.release()
    
    metadata = {
        'video_path': str(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'duration_seconds': duration,
        'frames_sampled': len(frame_indices),
        'faces_extracted': len(extracted),
        'faces_failed': failed,
        'frames': extracted
    }
    
    return metadata


def extract_audio(video_path, output_path, target_sr=16000):
    """
    Extract audio from video and save as WAV at 16kHz (for WavLM).
    
    Args:
        video_path: path to video file
        output_path: path to save audio WAV
        target_sr: target sample rate (16kHz for WavLM)
    
    Returns:
        dict with audio metadata
    """
    try:
        # Try loading audio directly with torchaudio
        waveform, sr = torchaudio.load(str(video_path))
    except Exception:
        # If torchaudio can't read the video format, use ffmpeg first
        import subprocess
        temp_wav = str(output_path) + ".temp.wav"
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn',  # no video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(target_sr),   # resample
            '-ac', '1',              # mono
            '-y',                    # overwrite
            temp_wav
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  ffmpeg failed for {video_path}")
            return None
        
        waveform, sr = torchaudio.load(temp_wav)
        os.remove(temp_wav)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to target sample rate
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Save
    torchaudio.save(str(output_path), waveform, target_sr)
    
    duration = waveform.shape[1] / target_sr
    
    return {
        'original_sr': sr,
        'target_sr': target_sr,
        'duration_seconds': duration,
        'num_samples': waveform.shape[1],
        'path': str(output_path)
    }


def process_dataset(input_dir, output_dir, max_frames=16, device='cuda'):
    """
    Process all videos in input_dir.
    
    Expected input structure:
        input_dir/real/*.mp4
        input_dir/fake/*.mp4
    
    Or just:
        input_dir/*.mp4  (for testing)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = []
    
    # Check for real/fake subdirectories
    for label in ['real', 'fake']:
        label_dir = input_dir / label
        if label_dir.exists():
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() in video_extensions:
                    videos.append((f, label))
    
    # Also check root directory
    if not videos:
        for f in sorted(input_dir.iterdir()):
            if f.suffix.lower() in video_extensions:
                videos.append((f, 'unknown'))
    
    if not videos:
        print(f"❌ No video files found in {input_dir}")
        print(f"   Expected: {input_dir}/real/*.mp4 and {input_dir}/fake/*.mp4")
        print(f"   Or:       {input_dir}/*.mp4")
        sys.exit(1)
    
    print(f"Found {len(videos)} videos to process")
    
    # Set up face detector
    print("Loading MTCNN face detector...")
    mtcnn = setup_face_detector(device)
    
    # Process each video
    all_metadata = []
    
    for video_path, label in tqdm(videos, desc="Processing videos"):
        video_name = video_path.stem
        video_out_dir = output_dir / label / video_name
        faces_dir = video_out_dir / "faces"
        audio_path = video_out_dir / "audio.wav"
        meta_path = video_out_dir / "metadata.json"
        
        # Skip if already processed
        if meta_path.exists():
            tqdm.write(f"  Skipping {video_name} (already processed)")
            continue
        
        os.makedirs(faces_dir, exist_ok=True)
        
        # Extract faces
        face_meta = extract_faces(video_path, faces_dir, mtcnn, max_frames)
        
        # Extract audio
        audio_meta = extract_audio(video_path, audio_path)
        
        # Combine metadata
        if face_meta and audio_meta:
            metadata = {
                'video_name': video_name,
                'label': label,
                'faces': face_meta,
                'audio': audio_meta
            }
            
            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            all_metadata.append(metadata)
            
            tqdm.write(
                f"  ✅ {video_name}: {face_meta['faces_extracted']}/{max_frames} faces, "
                f"{audio_meta['duration_seconds']:.1f}s audio"
            )
        else:
            tqdm.write(f"  ⚠️  {video_name}: processing failed")
    
    # Save summary
    summary = {
        'total_videos': len(videos),
        'processed': len(all_metadata),
        'labels': {label: sum(1 for _, l in videos if l == label) for label in set(l for _, l in videos)},
        'max_frames': max_frames
    }
    
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Processing complete!")
    print(f"   Total processed: {len(all_metadata)}/{len(videos)} videos")
    print(f"   Output saved to: {output_dir}")
    print(f"   Next step: run step3_clip_baseline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess videos for deepfake detection")
    parser.add_argument('--input', type=str, default='data/raw',
                        help='Input directory with videos (default: data/raw)')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory (default: data/processed)')
    parser.add_argument('--max_frames', type=int, default=16,
                        help='Frames to sample per video (default: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for face detection (default: cuda)')
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.max_frames, args.device)
