import tensorflow as tf
import numpy as np
import cv2  # Import OpenCV for image operations

def load_image(
        image_path,  # path of an image
        image_size=64,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
):
    if is_gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image

def save_batch_images(
        batch_images,   # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1,1),   # value range of the input batch images
        size_frame=None     # size of the image matrix, number of images in each row and column
):
    # transform the pixel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3], dtype=np.uint8)  # Initialize as uint8
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize each image in the batch
        image = (image * 255).astype(np.uint8)  # Convert to uint8
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    cv2.imwrite(save_path, frame)

# Example usage:
# image = load_image('image.jpg', image_size=64, is_gray=False)
# save_batch_images(batch_images, 'output.jpg', image_value_range=(-1, 1), size_frame=(4, 4))
