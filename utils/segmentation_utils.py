import torchvision.transforms as transforms
import cv2
import numpy as np
import torch

# import label_color_map
label_color_map = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128)  # tv/monitor
]

# define the torchvision image transforms
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)


def get_segment_labels(image, model, device):
    # transform the image to tensor and load into computation device
    image = transform(image).to(device)
    image = image.unsqueeze(0)  # add a batch dimension
    with torch.no_grad():
        outputs = model(image)
    return outputs


def draw_segmentation_map(outputs):
    print(f'outputs.shape={outputs.shape}')
    # print(f'outputs.squeeze={outputs.squeeze(dim=0)}')
    labels = torch.argmax(outputs.squeeze(dim=0), dim=0).detach().cpu().numpy()
    print(f'labels.shape={labels.shape}')
    print(f'labels={labels}')
    print(f'label max={np.max(labels)}')

    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    print(f'red_map.shape={red_map.shape}')

    for label_num in range(0, len(label_color_map)):
        index = labels == label_num
        red_map[index] = np.array(label_color_map)[label_num, 0]
        green_map[index] = np.array(label_color_map)[label_num, 1]
        blue_map[index] = np.array(label_color_map)[label_num, 2]
        # print(f'red_map[{index}]={red_map[index]}')

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    print(f'segmentation_map max={np.max(segmentation_map)}')
    return segmentation_map


def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta = 0.8  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image
