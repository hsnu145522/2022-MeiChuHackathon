import numpy as np
import cv2
import cv2.aruco as aruco
import glob



def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])

    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img+imgOut
    if drawId:
        cv2.putText(imgOut, str(id), (int(tl[0]),int(tl[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    return imgOut


def CalibrateCamera():
    ####---------------------- CALIBRATION ---------------------------
    # termination criteria for the iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # checkerboard of size (7 x 6) is used
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # iterating through all calibration images
    # in the folder
    images = glob.glob('calib_images/checkerboard/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chess board (calibration pattern) corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # if calibration pattern is found, add object points,
        # image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # Refine the corneirs of the detected corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs

def main():

    cap = cv2.VideoCapture(1)
    ret, mtx, dist, rvecs, tvecs = CalibrateCamera()
    imgAug = cv2.imread(glob.glob('calib_images/checkerboard/left01.jpg')[0])

    ###------------------ ARUCO TRACKER ---------------------------
    while (True):
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        #if ret returns false, there is likely a problem with the webcam/camera.
        #In that case uncomment the below line, which will replace the empty frame 
        #with a test image used in the opencv docs for aruco at https://www.docs.opencv.org/4.5.3/singlemarkersoriginal.jpg
        # frame = cv2.imread('./images/test image.jpg') 
        
        # operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            #for i in range(0, ids.size):
                # draw axis for the aruco markers
                #cv2.drawFrameAxes(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(frame, corners)

            for bbox, id in zip(corners, ids):
                frame = augmentAruco(bbox, id, frame, imgAug)
            # code to show ids of the marker found
            # strg = ''
            # for i in range(0, ids.size):
            #     strg += str(ids[i][0])+', '

            # cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (60,30), font, 1, (0,255,0),2,cv2.LINE_AA)

        # display the resulting frame
        cv2.imshow('Augmented Reality',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()