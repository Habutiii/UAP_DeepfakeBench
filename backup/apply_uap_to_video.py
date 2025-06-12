import os
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import signal


# import torch.nn.parallel
# import torch.utils.data
# import torch.optim as optim
import torchvision.transforms as transforms
from imutils import face_utils
from skimage import transform as trans

face_detector = None
face_predictor = None

# Define a global variable to track the current output file being processed
current_output_path = None


def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts

# all will output all found faces, otherwise only the biggest face
def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None, all=False):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    def get_face_content(face):
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        # landmark = predictor(cropped_face, face_align[0])
        # landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmarks, mask_face

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):

        if not all:
            # For now only take the biggest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            return get_face_content(face)

        else:
            # For all faces, we will return the first one
            # This is useful for video frames where multiple faces might be detected
            # and we want to process all of them
            cropped_faces = []
            all_landmarks = []
            mask_faces = []
            for face in faces:
                cropped_face, landmarks, mask_face = get_face_content(face)
                if cropped_face is not None:
                    cropped_faces.append(cropped_face)
                    all_landmarks.append(landmarks)
                    mask_faces.append(mask_face)
            if len(cropped_faces):
                # Return the all cropped face, landmark, and mask
                return cropped_faces, all_landmarks, mask_faces
    
    else:
        return None, None, None


def crop_center_square(image, center_x, center_y, size):
    h, w = image.shape[:2]
    half_size = size // 2

    # Ensure center_x and center_y are integers
    center_x = int(round(center_x))
    center_y = int(round(center_y))

    # Compute and cast bounding box to integers
    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, w)
    y2 = min(center_y + half_size, h)

    return image[int(y1):int(y2), int(x1):int(x2)]


def init_worker(predictor_path):
    global face_detector, face_predictor
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(predictor_path)

def extract_all_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    aligned_faces = []
    all_landmarks = []

    for face in faces:
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        aligned_faces.append(face)
        all_landmarks.append(landmarks)

    return faces, all_landmarks


def process_frame(args):
    
    global face_detector, face_predictor
    frame, rgb_frame, uap_original, uap_res, view_change = args
    # Extract aligned face and landmarks
    aligned_faces, all_landmarks = extract_all_face_dlib(face_detector, face_predictor, rgb_frame)

    frame_height, frame_width = frame.shape[:2]


    uap = uap_original.copy()
    # Paste UAP onto the frame
    # Find central points
    uap_center = uap_res // 2

    if len(aligned_faces):
        # apply UAP to all detected faces
        for i in range(0,len(aligned_faces)):
            aligned_face = aligned_faces[i]
            landmarks = all_landmarks[i]
            face_center_x, face_center_y = np.mean(landmarks, axis=0).astype(np.int)

            # Calculate top-left corner for UAP placement
            top_left_x = face_center_x - uap_center
            top_left_y = face_center_y - uap_center

            # print(f"Face center: ({face_center_x}, {face_center_y}), Top-left corner: ({top_left_x}, {top_left_y})")

            # Calculate bottom-right corner for UAP placement
            bottom_right_x = top_left_x + uap_res
            bottom_right_y = top_left_y + uap_res

            # Trim UAP if it goes outside the frame boundaries
            trim_top = max(0, -top_left_y)
            trim_left = max(0, -top_left_x)
            trim_bottom = max(0, bottom_right_y - frame_height)
            trim_right = max(0, bottom_right_x - frame_width)

            # Adjust the UAP dimensions based on the trimming
            trimmed_uap = uap[:, int(trim_top):int(uap.shape[1] - trim_bottom), int(trim_left):int(uap.shape[2] - trim_right)]

            # print(f"UAP shape after trimming: {trimmed_uap.shape}")

            # Adjust top-left corner after trimming
            top_left_x = max(top_left_x, 0)
            top_left_y = max(top_left_y, 0)


            # Debugging: Print shapes of frame and cropped UAP
            # print(f"Frame shape: {frame.shape}")
            # print(f"Cropped UAP shape: {trimmed_uap.shape}")

            # Adjust UAP shape to match the region of the frame
            # overlay_height = bottom_right_y - top_left_y
            # overlay_width = bottom_right_x - top_left_x

            # Convert frame to (C, H, W) format
            frame_chw = frame.transpose(2, 0, 1)

            # Ensure UAP is in uint8 format
            uap_chw = trimmed_uap.astype(np.float32)  # cast to uint8

            
            # Crop face
            face = crop_center_square(frame, face_center_x, face_center_y, uap_res)
            trimmed_face = face[:, int(trim_top):int(uap.shape[1] - trim_bottom), int(trim_left):int(uap.shape[2] - trim_right)]


            # Plot the first frame before and after UAP application
            if view_change:
                plt.figure(figsize=(10, 5))

                original_frame = frame.copy()

                # Plot original frame
                plt.subplot(2, 2, 1)
                plt.title(f"Original Frame {i}")
                plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                plt.axis("off")

                original_face = trimmed_face.copy()
                plt.subplot(2, 2, 3)
                plt.title("Original Face")
                plt.imshow(cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB))
                plt.axis("off")


            # Apply UAP to the same cropped face
            cropped_face_chw = face.transpose(2, 0, 1).astype(np.float32)  # Convert to (C, H, W) format and float32

            # Normalize the cropped face to [0, 1]
            cropped_face_normalized = cropped_face_chw / 255.0

            # Apply UAP to the normalized cropped face
            cropped_face_normalized += uap_chw

            # Clip values to valid range [0, 1]
            cropped_face_normalized = np.clip(cropped_face_normalized, 0, 1)

            # Revert the normalized cropped face back to [0, 255]
            uap_applied_face = (cropped_face_normalized * 255).astype(np.uint8).transpose(1, 2, 0)  # Convert back to (H, W, C) format



            # # Apply the UAP-applied face back to the original frame
            # top_left_x = int(face_center_x - uap_res // 2)
            # top_left_y = int(face_center_y - uap_res // 2)

            # # Ensure the coordinates are within the frame boundaries
            # top_left_x = max(0, top_left_x)
            # top_left_y = max(0, top_left_y)
            # bottom_right_x = min(frame_width, top_left_x + uap_res)
            # bottom_right_y = min(frame_height, top_left_y + uap_res)

            # Replace the region in the original frame with the UAP-applied face
            frame[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)] = uap_applied_face[:int(bottom_right_y - top_left_y), :int(bottom_right_x - top_left_x)]



            # Plot the first frame before and after UAP application
            if view_change:
                # Plot frame after UAP application
                plt.subplot(2, 2, 2)
                plt.title(f"Frame with UAP{i}")
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.axis("off")

                # Crop UAP-applied face

                plt.subplot(2, 2, 4)
                plt.title("UAP-Applied Face")
                plt.imshow(cv2.cvtColor(uap_applied_face, cv2.COLOR_BGR2RGB))
                plt.axis("off")


                plt.tight_layout()
                plt.show()

                # Compute the difference between the original frame and the final frame
                difference_frame = cv2.absdiff(original_frame, frame)

                # Plot the difference frame
                plt.subplot(1, 1, 1)
                plt.title("Difference Frame")
                plt.imshow(cv2.cvtColor(difference_frame, cv2.COLOR_BGR2RGB))
                plt.axis("off")

                plt.tight_layout()
                plt.show()

    return frame

def add_uap_to_video(config, pool):
    """
    Add UAP to a video based on facial landmarks.

    Args:
        config (dict): Configuration dictionary containing:
            - video_path (str): Path to the input video.
            - uap_path (str): Path to the UAP file (npy format).
            - output_path (str): Path to save the output video.
    """
    video_paths = config.get('video_paths')
    uap_path = config.get('uap_path')
    # output_path = config.get('output_path', './output_video.mp4')

    view_change = config.get('view_change', False)

    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(uap_path):
        raise FileNotFoundError(f"UAP file not found: {uap_path}")

    # Load UAP
    uap_original = np.load(uap_path)

    print(f"UAP shape: {uap_original.shape}")

    uap_res = uap_original.shape[2]  # Assuming UAP shape is (C, H, W)

    for video_path in video_paths:

        video_path = str(Path(video_path).resolve())

        output_path = str((Path(config.get('output_root_path', 'output_videos')) / (Path(uap_path).name + "_" + Path(video_path).name)).resolve())

        # Skip processing if the output video already exists
        if os.path.exists(output_path):
            print(f"Skipping video as output already exists: {output_path}")
            continue

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        first_frame_processed = False
        # Add progress bar for video processing
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Processing video frames")

        batch_size = 16
        frame_batch = []
        frame_info_batch = []

        face_detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(os.path.dirname(__file__), 'preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat')
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor file not found: {predictor_path}")
        face_predictor = dlib.shape_predictor(predictor_path)

        try:
            while True:
                ret, frame = cap.read()

                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_batch.append((frame, rgb_frame))

                # Process batch when full or when video ends
                if len(frame_batch) == batch_size or (not ret and len(frame_batch) > 0):
                    worker_args = [
                        (frame, rgb_frame, uap_original, uap_res, view_change)
                        for frame, rgb_frame in frame_batch
                    ]

                    processed_frames = pool.map(process_frame, worker_args)

                    for processed in processed_frames:
                        out.write(processed)
                        progress_bar.update(1)

                    frame_batch = []

                # Break only if no more frames and batch has been processed
                if not ret:
                    break

            cap.release()
            out.release()
            progress_bar.close()

            print(f"Video saved successfully at: {output_path}")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            if os.path.exists(output_path):
                print(f"Removing incomplete video file: {output_path}")
                os.remove(output_path)
            raise

        current_output_path = None  # Reset after successful processing

def handle_termination(signum, frame):
    global current_output_path
    if current_output_path and os.path.exists(current_output_path):
        print(f"Removing incomplete video file due to termination: {current_output_path}")
        os.remove(current_output_path)
    print("Script terminated forcefully.")
    exit(1)

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, handle_termination)


if __name__ == '__main__':

    video_root_path = '/mnt/data/Test Samples'
    # uap_path = 'robust_uap_meso4Inc_xception_resnet.npy'
    uap_path = 'robust_uap_ucf_xception.npy'
    # uap_path = 'robust_uap_xception_ucf_spsl_f3net.npy'
    output_root_path = 'output_videos'

    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    video_paths =[]


    for video in os.scandir(video_root_path):
        if video.is_file() and video.name.endswith('.mp4'):

            # check if filename end with _real.mp4
            if video.name.endswith('_real.mp4'):
                print(f"Skipping real video file: {video.name}")
                continue
            video_paths.append(video.path)
        else:
            print(f"Skipping non-video file: {video.name}")

    if not video_paths:
        raise ValueError("No video files found in the specified directory.")
            

    config = {
        'video_paths': video_paths,
        'uap_path': uap_path,
        'output_root_path': output_root_path,
        'view_change': False #True  # Set to True to view changes in frames for debugging   
    }


    mp.set_start_method('spawn', force=True)  # safer across platforms

    predictor_path = os.path.join(os.path.dirname(__file__), 'preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat')
    pool = mp.Pool(processes=8, initializer=init_worker, initargs=(predictor_path,))

    add_uap_to_video(config, pool)

    pool.close()
    pool.join()

