FROM python:3.9

WORKDIR /app

# ติดตั้ง libGL ที่ OpenCV ต้องใช้
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "flask_hand_gesture_api.py"]
