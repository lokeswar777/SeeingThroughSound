import cv2
import os
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS

# Load the BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)  # Capture every second

    count, saved = 0, 0
    prev_frame = None  # Store previous frame for movement detection

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for movement detection

            # Detect movement by comparing with previous frame
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                non_zero_count = np.count_nonzero(diff)
                movement_threshold = 5000  # Adjust threshold as needed
                
                if non_zero_count > movement_threshold:  # Movement detected
                    frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
                    cv2.imwrite(frame_path, frame)
                    saved += 1
            
            prev_frame = gray  # Update previous frame
        
        count += 1

    video.release()
    print(f"Extracted {saved} frames with movement.")

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def text_to_speech(text, output_path):
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)

# Main Execution
video_path = 'videoplayback (online-video-cutter.com).mp4'
frame_folder = '/kaggle/working/frames'

# Step 1: Extract frames with movement detection
extract_frames(video_path, frame_folder)

# Step 2: Generate captions and convert to speech (SORT FILES NUMERICALLY)
frame_files = sorted(os.listdir(frame_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))

for frame in frame_files:
    frame_path = os.path.join(frame_folder, frame)
    caption = generate_caption(frame_path)

    # Save audio file
    audio_path = os.path.join(frame_folder, f"{frame}.mp3")
    text_to_speech(caption, audio_path)
    
    print(f"{frame}: {caption} (Audio saved as {audio_path})")
