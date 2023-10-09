import cv2 as cv
import mediapipe as mp
import time

CIRCLE_SIZE = 7
COLOR = (255, 0, 0)

class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
    
    def find_hands(self, image, draw_hands=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return image

    def find_position(self, image, hand_number=0, draw_circle=False):
        lm_list = []

        if self.results.multi_hand_landmarks:
            hand_object = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand_object.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, cx, cy])
                if draw_circle:
                    cv.circle(image, (cx, cy), CIRCLE_SIZE, COLOR, cv.FILLED)
        
        return lm_list


def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    hand_detector = HandDetector()

    while True:
        success, image = cap.read()
        image = hand_detector.find_hands(image)
        lm_list = hand_detector.find_position(image)
        if len(lm_list) != 0:
            print(lm_list[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(
            image,
            str(int(fps)),
            (10, 70),
            cv.FONT_HERSHEY_PLAIN,
            3,
            COLOR,
            3
        )

        cv.imshow("Image", image)
        # A key event listener to exit when the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()