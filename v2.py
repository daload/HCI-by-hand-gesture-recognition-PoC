import logging
import time
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as f
import vgamepad as vg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import game_actions as ga
from game_actions import actions
from constants import targets, gestures

logging.basicConfig(format="[%(asctime)s] %(levelname)-8s %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class V2:
    mappings = {gestures[key]: actions[key] for (key) in gestures.keys() if key != 'right' and key != 'left'}
    turning_angle = 40

    @staticmethod
    def mark_flags(flag_arr, gesture):
        """
        Mark actions to perform

        Parameters
        ----------
        flag_arr : dict - Dictionary with the inputs to be marked
        gesture : string - Label of the gesture
        """
        if gesture in V2.mappings.keys():
            flag_arr[gesture] = 1

    @staticmethod
    def sim_input(gamepad, flags):
        """
        Simulate controller input (accelerate or brake)

        Parameters
        ----------
        gamepad : VX360Gamepad - Virtual controller
        flags : dict - Dictionary with the inputs (0 or 1)
        """
        for key in flags:
            V2.mappings[key](key, flags, gamepad)
            if gamepad:
                gamepad.update()

    @staticmethod
    def turn(deg, gp):
        """
        Simulate controller turning input

        Parameters
        ----------
        deg : float - Degrees of the angle formed
        gp : VX360Gamepad - Virtual controller
        """
        deg = (deg / V2.turning_angle) * -1
        if deg > 1:
            deg = 1
        if deg < -1:
            deg = -1
        ga.turn(deg, gp)
        gp.update()

    @staticmethod
    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray - input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((320, 320))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)

    @staticmethod
    def run(detector, threshold=0.5):
        """
        Run Gesture recognition model, draw bounding boxes on frame and simulate controller inputs

        Parameters
        ----------
        detector : TorchVisionModel - Gesture recognition model
        threshold : float - Confidence threshold
        """
        hands = mp.solutions.hands.Hands(
            model_complexity=0,
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7)

        gp = vg.VX360Gamepad()
        cap = cv2.VideoCapture(0)

        t1 = 0
        while cap.isOpened():
            gesture_flags = {key: 0 for key in V2.mappings.keys()}
            delta = (time.time() - t1)
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size, padded_size = V2.preprocess(frame)
                with torch.no_grad():
                    output = detector(processed_frame)[0]
                boxes = output["boxes"][:2]
                scores = output["scores"][:2]
                labels = output["labels"][:2]
                results = hands.process(frame[:, :, ::-1])
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                            mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1))

                if results:
                    if results.multi_hand_landmarks:
                        if len(results.multi_hand_landmarks) >= 2:
                            h, w, p = frame.shape
                            if results.multi_hand_landmarks[0].landmark[0].x < results.multi_hand_landmarks[1].landmark[
                                0].x:
                                lm1 = (int(results.multi_hand_landmarks[0].landmark[0].x * w),
                                       int(results.multi_hand_landmarks[0].landmark[0].y * h))
                                lm2 = (int(results.multi_hand_landmarks[1].landmark[0].x * w),
                                       int(results.multi_hand_landmarks[1].landmark[0].y * h))
                            else:
                                lm1 = (int(results.multi_hand_landmarks[1].landmark[0].x * w),
                                       int(results.multi_hand_landmarks[1].landmark[0].y * h))
                                lm2 = (int(results.multi_hand_landmarks[0].landmark[0].x * w),
                                       int(results.multi_hand_landmarks[0].landmark[0].y * h))
                            angle = np.arctan2(lm2[1] - lm1[1], lm2[0] - lm1[0])
                            angle_deg = angle * 180 / np.pi
                            cv2.line(frame, lm1, lm2, [255, 0, 255], 2)

                            V2.turn(angle_deg, gp)
                        else:
                            V2.turn(0, gp)

                for i in range(min(2, len(boxes))):
                    if scores[i] > threshold:
                        width, height = size
                        padded_width, padded_height = padded_size
                        scale = max(width, height) / 320

                        padding_w = abs(padded_width - width) // (2 * scale)
                        padding_h = abs(padded_height - height) // (2 * scale)

                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)
                        hand_gesture = targets[int(labels[i])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                        cv2.putText(frame, hand_gesture, (x1, y1 - 10),
                                    FONT, 2, (0, 0, 255), thickness=3)
                        V2.mark_flags(gesture_flags, hand_gesture)

                V2.sim_input(gp, gesture_flags)
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}", (30, 30), FONT, 1, COLOR, 2)

                cv2.imshow('Frame', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()
