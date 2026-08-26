"""Standalone hardware test: webcam -> YOLO tracking -> clickable selection -> crop."""
from __future__ import annotations
import os
import time
import cv2
from src.crop import crop_detection
from src.detector import ObjectDetector
from src.selection import Selection

ROOT=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(ROOT,"outputs","crops")
os.makedirs(OUT,exist_ok=True)
selection=Selection()
detections=[]


def main():
    global detections
    detector=ObjectDetector()
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam.")
    window="VizoLearn Core Test"
    cv2.namedWindow(window)

    def on_mouse(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN: return
        match=selection.select_at(detections,x,y)
        if match: print(f"Selected: {match['name']} | track={match['track_id']} | confidence={match['confidence']:.2f}")
        else: print("No detected object at that point.")
    cv2.setMouseCallback(window,on_mouse)
    print("Click a detected object. S=save crop, C=clear, Q=quit.")
    try:
        while True:
            ok,frame=cap.read()
            if not ok: raise RuntimeError("Could not read webcam frame.")
            detections=detector.detect(frame)
            current=selection.current(detections)
            for d in detections:
                x1,y1,x2,y2=d['box']
                chosen=current is not None and d.get('track_id')==current.get('track_id')
                color=(0,0,255) if chosen else (0,220,100)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if chosen else 2)
                label=f"{d['name']} ID:{d['track_id']} {d['confidence']:.2f}"
                cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
            cv2.imshow(window,frame)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            if key==ord('c'): selection.clear()
            if key==ord('s') and current is not None:
                crop=crop_detection(frame,current,padding_ratio=0.02)
                if crop is not None:
                    path=os.path.join(OUT,f"{current['name']}_{current['track_id']}_{int(time.time())}.jpg")
                    cv2.imwrite(path,crop)
                    print("Saved:",path)
    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()