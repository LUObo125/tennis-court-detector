import time
from court_detection import CourtDetector
import cv2 


def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height



def read_video(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames



def main():
    input_video_path = "input_video/input_video.mp4"
    output_folder_has = "out_folder/has_court"
    output_folder_no = "out_folder/no_court"
    # time counter
    total_time = 0
    count_no = 636
    count_has = 2137

    #court detect flag
    court_det_flag = False
    court_inproc_flag = False

    video = cv2.VideoCapture(input_video_path)
    fps, length, v_width, v_height = get_video_properties(video)
    video_frames = read_video(video)
    court_detector = CourtDetector()



    for i, frame in enumerate(video_frames):
        if i < 585:
            continue
        start_time = time.time()
        if i == 0 or court_det_flag == False:
            try:
                court_inproc_flag = court_detector.detect(frame)
            except:
                court_inproc_flag = False

        if court_inproc_flag:
            court_det_flag = court_detector.track_court(frame)
        if court_det_flag:
            cv2.imwrite(f"{output_folder_has}/{count_has}.jpg", frame)     # 保存帧
            count_has += 1
        else:
            cv2.imwrite(f"{output_folder_no}/{count_no}.jpg", frame)     # 保存帧
            count_no += 1

        total_time += (time.time() - start_time)
        print('Processing frame %d/%d  FPS %04f' % (i, length, i / total_time), '\r', end='')
        if not i % 100:
            print('')
    print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    print('Processing completed')

if __name__ == "__main__":
    main()