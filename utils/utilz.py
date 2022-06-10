import numpy as np

from utils.cv2_imshow import cv2_imshow


def check_output(arr):
    arr[arr < 1] = 0.0
    arr[arr > 0] = 1.0
    output = arr.astype('uint8')
    output = output * 255
    print(f'** output={output}')
    print(f'output max={np.max(output)}')
    c, w, h = output.shape
    print(c, w, h)
    output = np.reshape(output, (w, h, c))

    cv2_imshow(output, 'check output')
