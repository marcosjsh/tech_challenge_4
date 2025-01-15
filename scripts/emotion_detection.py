import os
import cv2
import os
from deepface import DeepFace
from tqdm import tqdm


def detect_emotions(video_path, output_path):
    if os.path.exists(video_path) == False:
        return False, None, f"File not found at path: {video_path}"

    video_capture = cv2.VideoCapture(video_path)
    if video_capture.isOpened() == False:
        return False, None, "Erro ao abrir o vídeo."
    
    print(f"Iniciando análise do vídeo: {video_path}")

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, codec, fps, (width, height))

    for _ in tqdm(range(frame_count), desc="Analisando expressões faciais"):
        ret, frame = video_capture.read()

        if not ret:
            break

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in result:
            x, y, width, height = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            dominant_emotion = face['dominant_emotion']
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        output.write(frame)

    video_capture.release()
    output.release()
    cv2.destroyAllWindows()
    
    return True, output_path, "Análise de expressões faciais realizado com sucesso!"