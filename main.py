import cv2 as cv
from classes import *

camera = cv.VideoCapture(1)
p = Program("configs\\config.json", "configs\\MultiMatrix.npz", [1040, 580])
cues = []

while True:

    # Get camera frame
    _, frame = camera.read()
    undisCamera = p.undistort(frame)
    cv.imshow("undisCamera", undisCamera)

    # Detect Markers


    if p.state == 0:
        markers = p.detectMarkers(undisCamera)
        markersDisplay = undisCamera.copy()
        p.displayMarkers(markersDisplay, [], markers)

        if all(markers.get(x) for x in range(4)):
            # Calculate prospective Warp
            markerAngles = p.prospectiveCalculate(markersDisplay, markers, [0, 1, 2, 3])
            p.state = 1


    if p.state == 1:
        # Warp Prospective
        prospectiveWarp = p.prospectiveWarp(undisCamera, markerAngles)
        markers = p.detectMarkers(prospectiveWarp)

        # Detect Balls
        ballList = p.detectBalls(prospectiveWarp, True)
        [ball.display(prospectiveWarp) for ball in ballList if isinstance(ball, Ball)]

        #! Cues
        cues = p.getCuePositions(markers, [[17, 37], [41, 38]], cues)
        for cue in cues:
            cue: Cue
            if cue:
                cue.display(prospectiveWarp)
                p.simulator.process(ballList, prospectiveWarp, cue)

        cv.imshow("prospectiveWarp", prospectiveWarp)

    cv.imshow("markersDisplay", markersDisplay)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        p.takeImage(prospectiveWarp)

camera.release()
cv.destroyAllWindows()