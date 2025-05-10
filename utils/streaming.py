from threading import Thread
import cv2


class VStream:
    def __init__(self, stream_id=0, width=1024, height=512):
        self.width = width
        self.height = height
        stream = cv2.VideoCapture()
        stream.open(stream_id)
        stream.set(cv2.CAP_PROP_POS_MSEC, 64_000)
        self.stream = stream
        self.ret, self.frame = self.stream.read()
        self.t = Thread(target=self.update, args=())
        self.deamon = True
        self.stopped = False

    def __del__(self):
        self.stream.release()

    def start(self):
        self.stopped = False
        self.t.start()

    def read(self):
        return self.frame

    def update(self):
        while True:
            if self.stopped:
                break
            status, frame = self.stream.read()
            if status:
                self.frame = cv2.resize(
                    frame, (self.width, self.height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            else:
                self.stopped = True
                break

        self.stream.release()

    def stop(self):
        self.stopped = True
