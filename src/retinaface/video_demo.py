import os
import os.path as osp
import sys
import argparse

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.detect_face import FaceDetector
from app.view_util import draw_bbox, draw_landmark, view_image
from app.model_config import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RetinaFace')
    # general
    parser.add_argument('--input',
                        help='Input video path',
                        type=str)
    parser.add_argument('--output',
                        default='tmp.avi',
                        help='Output path to save video')
    parser.add_argument('-d', '--debug', action='store_true',
        help='Whether to debug.')
    parser.add_argument('-m','--max-num', type=int, default=200,
        help='max number of frames to detect.')
    parser.add_argument('-s', '--start-frame', type=int, default=0,
        help='Start frame')
    args = parser.parse_args()

    assert osp.exists(args.input), f'File not exists: {args.input}'
    save_dir, _ = os.path.split(args.output)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    detector = FaceDetector(cfg.MODEL_RETINAFACE)
        
    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output,fourcc, 20.0, (1280,960))

    counter = 0
    end_idx = args.start_frame + args.max_num
    while(cap.isOpened() and counter < end_idx):
        ret, frame = cap.read()
        if ret==True:
            counter += 1
            if counter < args.start_frame:
                continue
            # view_image(frame, name='original', wait_key=True)
            bboxes, landmarks = detector(frame)
            if bboxes is not None and bboxes.size:
                for bb in bboxes:
                    frame = draw_bbox(frame, bb)
            if landmarks is not None and landmarks.size:
                for pts in landmarks:
                    frame = draw_landmark(frame, pts)
            if args.debug:
                view_image(frame, 'retinaface', wait_key=True, win_width=1280, win_height=960)

            # write the flipped frame
            frame = cv2.flip(frame,0)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()