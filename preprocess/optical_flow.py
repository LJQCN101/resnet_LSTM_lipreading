import cv2, numpy as np, os
from xinshuo_io import mkdir_if_missing, load_image, load_list_from_folder, save_image, fileparts
from xinshuo_images import rgb2gray
from flow_vis import flow_to_color

data_dir = '/media/xinshuo/Data/Datasets/LRD/LRW/'
images_dir = os.path.join(data_dir, 'centered122_rgb_images/ABOUT/train/ABOUT_00001')

save_dir = os.path.join(data_dir, 'centered122_flow/ABOUT/train/ABOUT_00001'); mkdir_if_missing(save_dir)

image_list, num_images = load_list_from_folder(images_dir)
print('number of images loaded is %d' % num_images)


test = cv2.imread(image_list[0])
# print(test.dtype)
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# print(test.dtype)
# print(test)
# zxc


frame1 = load_image(image_list[0])
prvs = rgb2gray(frame1)         # uint8

# print(prvs)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

image_index = 1
while(1):
    image_path = image_list[image_index]
    _, filename, _ = fileparts(image_path)

    frame2 = load_image(image_path)
    next = rgb2gray(frame2)
    # save_image(next, '/home/xinshuo/aa.jpg')
    # zxc
    # cv2.imshow('asd', next)
    # k = cv2.waitKey(30)
    # zxc

    # ret, frame2 = cap.read()
    # next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow_uv = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    rgb = flow_to_color(flow_uv, convert_to_bgr=False)

    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # hsv[...,0] = ang * 180 / np.pi / 2
    # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow('frame2', rgb)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
        # break
    # elif k == ord('s'):
    save_path_tmp = os.path.join(save_dir, filename+'.jpg')
    save_image(rgb, save_path=save_path_tmp)
    # cv2.imwrite('opticalfb.png', frame2)
    # cv2.imwrite('opticalhsv.png', rgb)
    prvs = next
    image_index += 1
    # zxc

cap.release()
cv2.destroyAllWindows()