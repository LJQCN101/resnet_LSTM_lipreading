# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import imageio
# imageio.plugins.ffmpeg.download()
import torchvision.transforms.functional as functional, torchvision.transforms as transforms, torch, cv2
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
from xinshuo_visualization import save_image
from xinshuo_images import image_bgr2rgb
from xinshuo_miscellaneous import is_path_exists

frist_crop_size = 122
final_crop_size = 112
num_frames = 29

def load_video(filename):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    vid = imageio.get_reader(filename, 'ffmpeg')
    frames = []
    for i in range(0, num_frames):
        image = vid.get_data(i)             # uint8 256 x 256 x 3
        # image = image[116:212, 80:176, :]            # 96 x 96
        # image = image[108:220, 72:184, :]            # 112 x 112
        image = image[103:225, 67:189, :]            # 122 x 122
        image = functional.to_tensor(image)
        frames.append(image)
    return frames


def load_video_opencv(filename, debug=True):
    '''
    if the VideoCapture does not work, uninstall python-opencv and reinstall the newest version
    '''
    if debug: assert is_path_exists(filename), 'the input video file does not exist'
    cap = cv2.VideoCapture(filename)
    frame_id = 0
    frames = []

    while True:
        ret, image = cap.read()
        if not ret: break

        image = image_bgr2rgb(image)
        image = functional.to_tensor(image)
        frames.append(image)
        frame_id += 1
        if frame_id == num_frames: break

    # frames = np.array(frames)
    return frames

def bbc(vidframes, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(3, num_frames, final_crop_size, final_crop_size)
    # croptransform = transforms.CenterCrop(crop_size)
    if augmentation:
        crop = StatefulRandomCrop((frist_crop_size, frist_crop_size), (final_crop_size, final_crop_size))
        flip = StatefulRandomHorizontalFlip(0.5)
        croptransform = transforms.Compose([crop, flip])
    else: croptransform = transforms.CenterCrop(final_crop_size)

    for frame_index in range(0, num_frames):
        overall_transform = transforms.Compose([transforms.ToPILImage(), croptransform,
        # overall_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), croptransform,
            transforms.ToTensor(), transforms.Normalize([0.4161, ], [0.1688, ])])
        result = overall_transform(vidframes[frame_index])

        # print(result.shape)
        temporalvolume[0][frame_index] = result[0, :, :]
        temporalvolume[1][frame_index] = result[1, :, :]
        temporalvolume[2][frame_index] = result[2, :, :]

        # visualization
        # overall_transform2 = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale()]) 
        # overall_transform2 = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), croptransform]) 
        # result2 = overall_transform2(vidframes[frame_index])
        # save_image(result2, '/home/xinshuo/test%d.jpg' % frame_index)

    # zxc
    return temporalvolume