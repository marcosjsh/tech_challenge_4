import os
from emotion_detection import detect_emotions


#Aula 2.2 temos detecção de emoção com o quadradinho de reconhecimento de face

if __name__ == '__main__':
    raw_video_path = 'assets/origin_video.mp4'
    emotions_output_video_path = 'assets/emotions_video.mp4'

    success, output_path, message = detect_emotions(raw_video_path, emotions_output_video_path)
    
    print(message)
    
    if success:
        print(output_path)
    