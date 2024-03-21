from collections import defaultdict
from ultralytics import YOLO
import json, math
import numpy as np
import cv2 as cv

class Marker:
    def __init__(self, id, angles) -> None:
        self.id = id[0]
        self.angle1 = angles[0]
        self.angle2 = angles[1]
        self.angle3 = angles[2]
        self.angle4 = angles[3]
        self.angles = [np.array(angles, np.int32)]
        self.angle1N = [int(x) for x in self.angle1]
        self.angle2N = [int(x) for x in self.angle2]
        self.angle3N = [int(x) for x in self.angle3]
        self.angle4N = [int(x) for x in self.angle4]
        self.anglesN = [[int(x) for x in inner] for inner in angles]
    
    @property
    def center(self):
        return ((self.centerUp[0] + self.centerDown[0]) / 2, (self.centerUp[1] + self.centerDown[1]) / 2)
    @property
    def centerUp(self):
        return ((self.angle1N[0] + self.angle2N[0]) / 2, (self.angle1N[1] + self.angle2N[1]) / 2)
    @property
    def centerDown(self):
        return ((self.angle3N[0] + self.angle4N[0]) / 2, (self.angle3N[1] + self.angle4N[1]) / 2)

    def __str__(self) -> str:
        return f"id: {self.id}, angles: {self.angles}, anglesN: {self.angles}"


class Ball:
    def __init__(
        self, id: int, xyxy: (float, float, float, float), conf: int, radius: int = 19
    ):
        self.id: int = id
        self.xyxy: (float, float, float, float) = xyxy
        self.conf: int = conf
        self.radius: int = radius

    def __str__(self) -> str:
        return f"Id: {self.id} | xyxyI: {self.xyxyI} | whI: {self.whI} | centerI: {self.centerI}"

    @property
    def xyxyI(self) -> (int, int, int, int):
        x1, y1, x2, y2 = self.xyxy
        return (int(x1), int(y1), int(x2), int(y2))

    @property
    def wh(self) -> (float, float):
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1, y2 - y1)

    @property
    def whI(self) -> (int, int):
        w, h = self.wh
        return int(w), int(h)

    @property
    def center(self) -> (float, float):
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def centerI(self) -> (int, int):
        x, y = self.center
        return int(x), int(y)

    @property
    def confI(self) -> (int, int):
        return math.ceil((self.conf * 100)) / 100
    
    
    def display(self, frame):
        center = self.centerI

        BALL_TYPES = {
            # example: [(rgb), (outline/txt_rgb) ball_half/full, offset_txt]
            0: [(255, 255, 255), (0, 0, 0), False, False],
            1: [(0, 255, 255), (0, 0, 0), False, False],
            2: [(224, 34, 40), (255, 255, 255), False, False],
            3: [(0, 0, 255), (0, 0, 0), False, False],
            4: [(74, 17, 21), (255, 255, 255), False, False],
            5: [(29, 152, 240), (0, 0, 0), False, False],
            6: [(20, 99, 13), (0, 0, 0), False, False],
            7: [(17, 13, 89), (255, 255, 255), False, False],
            8: [(0, 0, 0), (255, 255, 255), False, False],
            9: [(0, 255, 255), (0, 0, 0), True, False],
            10: [(224, 34, 40), (255, 255, 255), True, True],
            11: [(0, 0, 255), (0, 0, 0), True, True],
            12: [(74, 17, 21), (255, 255, 255), True, True],
            13: [(29, 152, 240), (0, 0, 0), True, True],
            14: [(20, 99, 13), (0, 0, 0), True, True],
            15: [(17, 13, 89), (255, 255, 255), True, True],
        }

        ball_type = BALL_TYPES[self.id]

        if ball_type[3]:
            txt_pos = center[0] - 16, center[1] + 6
        else:
            txt_pos = center[0] - 6, center[1] + 6

        cv.circle(frame, center, 19, ball_type[0], -1)
        cv.circle(frame, center, 19, ball_type[1], 1)
        cv.putText(
            frame,
            str(self.id),
            txt_pos,
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            ball_type[1],
            1,
            cv.LINE_AA,
        )


class Cue:

    def __init__(self, point_1: (float, float), point_2: (float, float), color):

        ## point_1 / point_2 Servono a trovare la rotazione e posizione nel frame della mazza
        self.point_1: (float, float) = point_1 #punto più vicino alla punta
        self.point_2: (float, float) = point_2 #punto più lontano dalla punta
        self.color = color


    def __str__(self) -> str:
        return f"x: {self.point_1I}, y: {self.point_1I} | ↗ {self.direction_vector_normalized}"


    @property
    def direction_vector(self):
        return np.array(self.point_2) - np.array(self.point_1)

    @property
    def direction_vector_normalized(self):
        direction_vector = self.direction_vector
        return direction_vector / np.linalg.norm(direction_vector)

    @property
    def point_1I(self) -> (int, int):
        x, y = self.point_1
        return int(x), int(y)
    
    @property
    def point_2I(self) -> (int, int):
        x, y = self.point_2
        return int(x), int(y)


    def display(self, frame) -> None:

        # Lengths on both sides (im mm/px)
        length_lower = 80
        length_higher = 1000

        direction_vector_normalized = self.direction_vector_normalized

        # Calculate the endpoints of the lines
        lower_endpoint = tuple(np.array(self.point_1I) + direction_vector_normalized * length_lower)
        higher_endpoint = tuple(np.array(self.point_2I) - direction_vector_normalized * length_higher)

        # Convert to ints
        lower_endpoint = (int(lower_endpoint[0]), int(lower_endpoint[1]))
        higher_endpoint = (int(higher_endpoint[0]), int(higher_endpoint[1]))

        # Draw the lines
        cv.line(frame, self.point_2I, lower_endpoint, self.color, 2)
        cv.line(frame, self.point_1I, higher_endpoint, self.color, 2)
        
    
    def simulate(self, frame) -> None:
        pass


class Simulator:
    
    def __init__(self, max_rimbalzi: int = 2, table_size= (1040, 588), offset= (35, 35)) -> None:
        self.max_rimbalzi = max_rimbalzi
        self.table_size = table_size
        self.offset = offset

    def get_direction_vector_normalized_2_points(self, point_1, point_2):
        direction_vector = np.array(point_2) - np.array(point_1)
        return direction_vector / np.linalg.norm(direction_vector)

    def process(self, ball_list, frame, cue) -> None:

        point_2I, direction = cue.point_2I, cue.direction_vector_normalized
        x, y = point_2I
        pos = np.array([x, y], dtype=np.float32)
        circle_radius = []
        rimbalzi_muro = 0
        collisioni_calcolate = 0
        holes = [
            (60, 53, 30), (60, 520, 30),
            (515, 45, 30), (515, 545, 30),
            (980, 58, 30), (983, 520, 30)
        ]
        passato_per_bianca = False

        circle_centers = np.array([ball.centerI for ball in ball_list])
        for ball in ball_list:
            if ball.id == 0:
                circle_radius.append(7)
            else:
                circle_radius.append(35)
        circle_radius = np.array(circle_radius)

        # Simulare il percorso con rimbalzi
        while (self.max_rimbalzi > rimbalzi_muro) and (collisioni_calcolate < 3):

            pos = pos + direction

            # Check if the path intersects with any of the holes
            for circle_x, circle_y, r in holes:
                if np.sqrt((pos[0] - circle_x)**2 + (pos[1] - circle_y)**2) < r:
                    return frame

            #----------------------------------------------------------------------------------#
            if ball_list != []:
                intersections = np.sqrt((pos[0] - circle_centers[:, 0])**2 + (pos[1] - circle_centers[:, 1])**2)
                intersect_indices = np.where(intersections < circle_radius)

                if len(intersect_indices[0]) > 0:
                    hit_index = intersect_indices[0][0]
                    hit_position = circle_centers[hit_index]
                    if circle_radius[hit_index] == 7:
                        passato_per_bianca = True
                    else:
                        cv.circle(frame, (int(hit_position[0]), int(hit_position[1])), 23, (0,0,0), 2)
                        cv.circle(frame, (int(hit_position[0]), int(hit_position[1])), 3, (0,0,0), -1)
                        cv.circle(frame, (int(pos[0]), int(pos[1])), 5, (255, 255, 255), -1)
                        direction = self.get_direction_vector_normalized_2_points((pos[0], pos[1]), (hit_position[0], hit_position[1]))
                        pos = pos + (direction*100)
                        collisioni_calcolate += 1
            #----------------------------------------------------------------------------------#

            # Gestire i rimbalzi sui lati del tavolo
            for i in range(2):
                if pos[i] < self.offset[i] or pos[i] > self.table_size[i] - self.offset[i]:
                    direction[i] = -direction[i]
                    rimbalzi_muro += 1

            #colore linea
            color = (0, 255, 0) if passato_per_bianca == True else (0, 0, 255)


            # Disegnare la linea della traiettoria
            cv.line(frame, (int(x), int(y)), (int(pos[0]), int(pos[1])), color, 2)

            # Aggiornare le coordinate x e y
            x, y = pos


class Program:

    def __init__(
        self, configPath: str, calibDataPath: str, simulationDefinition: (int, int)
    ) -> None:
        self.configPath = configPath
        self.config = self.loadConfig()
        self.calib_data = np.load(calibDataPath)
        self.simulationDefinition = simulationDefinition
        self.ballModel = YOLO(f'models\\pool_v{self.config["model"]}.pt')
        self.simulator = Simulator()
        self.state = 0


    def loadConfig(self) -> dict:
        with open(self.configPath, "r") as f:
            return json.load(f)
        
    def saveConfig(self) -> None:
        with open(self.configPath, 'w') as json_file:
            json.dump(self.config, json_file)

    def undistort(self, frame):
        camMatrix, distCoef = self.calib_data["camMatrix"], self.calib_data["distCoef"]
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(
            camMatrix, distCoef, (w, h), 1, (w, h)
        )
        dst = cv.undistort(frame, camMatrix, distCoef, None, newcameramtx)
        x, y, w, h = roi
        return dst[y : y + h, x : x + w]

    def detectMarkers(self, frame) -> dict:
        markerCorners, markerIDs, _ = cv.aruco.detectMarkers(
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY),
            cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50),
            parameters=cv.aruco.DetectorParameters_create(),
        )
        dict = {}
        if markerCorners:
            for corners, id in zip(markerCorners, markerIDs):
                dict[id[0]] = Marker(id, corners.reshape(-1, 2).tolist())
            return dict
        else:
            return {}

    def displayMarkers(self, frame, markerList: list or None, markers: dict) -> None:
        if not markerList:
            markerList = markers
        for markerId in markerList:
            m: Marker = markers.get(markerId)
            cv.putText(
                frame,
                str(m.id),
                m.angle1N,
                cv.FONT_HERSHEY_SIMPLEX,
                1.3,
                (25, 50, 50),
                1,
                cv.LINE_AA,
            )
            cv.polylines(frame, m.angles, True, (0, 0, 255), 1, cv.LINE_AA)

    def prospectiveCalculate(self, frame, markers, ids: list) -> list:
        points = []

        marker0: Marker = markers[ids[0]]
        marker1: Marker = markers[ids[1]]
        marker2: Marker = markers[ids[2]]
        marker3: Marker = markers[ids[3]]

        points = [marker0.angle1N, marker1.angle1N, marker2.angle1N, marker3.angle1N]

        cv.polylines(
            frame, [np.array(points, np.int32)], True, (0, 255, 0), 4, cv.LINE_AA
        )

        return [marker2.angle1N, marker1.angle1N, marker3.angle1N, marker0.angle1N]

    def prospectiveWarp(self, frame, markerAngles):
        fDef = self.simulationDefinition
        pos = np.float32([[0, 0], [0, fDef[1]], [fDef[0], 0], [fDef[0], fDef[1]]])
        matrix = cv.getPerspectiveTransform(np.float32(markerAngles), pos)
        prospectiveWarp = cv.warpPerspective(frame, matrix, fDef)
        return prospectiveWarp

    def detectBalls(self, frame, filter: bool) -> [Ball]:
        
        frame1 = frame.copy()
        self.hideHoles(frame1)
        detections = self.ballModel.predict(
            frame1, conf=0.1, device="0", iou=0.5, verbose=False
        )

        if detections:
            real_classes = [0, 1, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8, 9]

            for r in detections:
                ball_list = []

                for box in r.boxes:
                    conf = float(box.conf[0])
                    if math.ceil((conf * 100)) / 100 >= 0.1:
                        ball = Ball(real_classes[int(box.cls[0])], box.xyxy[0], conf)
                        if (ball.whI[0] < 50) and (ball.whI[1] < 50):
                            ball_list.append(ball)

            if filter is True:
                max_scores = defaultdict(float)
                result_dict = defaultdict(dict)

                # Iterazione attraverso i dati e mantenimento del punteggio massimo per ogni classe
                for ball in ball_list:
                    ball: Ball

                    class_num = ball.id
                    if ball.conf > max_scores[class_num]:
                        max_scores[class_num] = ball.conf
                        result_dict[class_num] = ball

                return list(result_dict.values())

            else:
                return ball_list
        return []

    def hideHoles(self, frame) -> None:
        color = (0, 104, 20)
        #sinistra
        cv.circle(frame, (60,53), 40, color, -1)
        cv.circle(frame, (60,520), 40, color, -1)
        #centro
        cv.circle(frame, (515,45), 40, color, -1)
        cv.circle(frame, (515,545), 40, color, -1)
        #destra
        cv.circle(frame, (980,58), 40, color, -1)
        cv.circle(frame, (983,520), 40, color, -1)

    def getCuePositions(frame, markers: {}, cueMarkers: [[int, int], [int, int]], oldCues) -> [Cue]:
        
        cues, newCues = [], []
        
        for cueMarker in cueMarkers:

            marker1: Marker = markers.get(cueMarker[0])
            marker2: Marker = markers.get(cueMarker[1])

            if marker1 and marker2:
                cues.append(Cue(marker1.center, marker2.center, (0, 0, 0)))

            elif marker1:
                cues.append(Cue(marker1.centerUp, marker1.centerDown, (0, 0, 0)))

            elif marker2:
                cues.append(Cue(marker2.centerUp, marker2.centerDown, (0, 0, 0)))

            else:
                cues.append(None)
        

        return cues

    def takeImage(self, frame) -> None:

        ball_list = self.detectBalls(frame, True)
        """
        {
            "predictions": [
                {
                    "x": 599.0,
                    "y": 395.0,
                    "width": 46.0,
                    "height": 38.0,
                    "confidence": 0.8305338621139526,
                    "class": "10",
                    "image_path": "imgs\\imgs\\16.png",
                    "prediction_type": "ObjectDetectionModel"
                }
            ],
            "image": {
                "width": "1040",
                "height": "580"
            }
        }
        """

        annotations = {
            "predictions": [],
            "image": {
                "width": "1040",
                "height": "580"
            }
        }
        for ball in ball_list:
            ball: Ball
            x, y = ball.centerI
            w, h = ball.whI
            annotations["predictions"].append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": ball.conf,
                    "class": ball.id,
                    "image_path": f"imgs\\imgs\\{self.config['img']}.png",
                    "prediction_type": "ObjectDetectionModel"
                }
            )

        # Save
        cv.imwrite(f"imgs\\captures\\{self.config['img']}.png", frame)
        with open(f"imgs\\annotations\\{self.config['img']}.json", 'w') as f:
            json.dump(annotations, f, indent = 4)
        print(f"Saved image as {self.config['img']}")

        self.config['img'] += 1
        self.saveConfig()