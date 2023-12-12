"""
Code adapted from  

https://docs.ultralytics.com/modes/track/#tracking-arguments
https://docs.openvino.ai/2023.0/notebooks/notebooks.html
"""

import collections
import numpy as np
from IPython import display
import cv2
import openvino as ov
from notebook_utils import VideoPlayer
from yolo_pipeline import draw_results

core = ov.Core()


# Main processing function to run object detection.
def ov_video_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0,
                              detector=None, labels=None):
    """
    By default, the primary webcam is set with `source=0`. If you have multiple webcams,
    each one will be assigned a consecutive number starting at 0.
    Set `flip=True` when using a front-facing camera. Some web browsers, especially Mozilla
    Firefox, may cause flickering. If you experience flickering, set `use_popup=True`. """

    player = None

    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:

            # Get the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA)

            # model expects RGB image, while video capturing in BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector.frame = frame

            # detects
            input_image, detections, time = detector.detect()
            image_with_boxes = draw_results(detections[0], input_image, labels=labels)
            frame = image_with_boxes

            processing_times.append(time)

            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=f_width / 1200,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname="Press 27 to Exit", mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                im = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(im)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()


def yolo_video_track(source=None, yolo_model=None):
    """
    This function is adapted from ultraliytics
    https://docs.ultralytics.com/modes/track/#tracking-arguments
    """

    # Open the video file

    cap = cv2.VideoCapture(source)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = yolo_model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()