import cv2
import threading
import queue
import time

class CameraStream:
    def __init__(self, cam_id, source, fps_limit=30):
        self.cam_id = cam_id
        self.source = source
        self.fps_limit = fps_limit
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _reader_loop(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            if not cap.isOpened():
                print(f"[{self.cam_id}] Reconnecting to {self.source}...")
                cap.open(self.source)
                time.sleep(2)
                continue
                
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1)
                continue
                
            # Keep queue fresh with latest frame
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
            
            time.sleep(1.0 / self.fps_limit)
            
        cap.release()

    def read(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None
