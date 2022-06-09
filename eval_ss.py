import argparse
import numpy as np
# from utils import segmentation_utils
import cv2
import torch
from PIL import Image

from utils import segmentation_utils
from utils.cv2_imshow import cv2_imshow
from utils.utilz import check_output

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input image',
    # default='../potato/set37/images/potato0.jpg'
    # default='../images/dog_person.jpg'
    default='./CrackForest/Images/001.jpg'
    )
parser.add_argument('-m', '--model', dest='model_path', help='path to model', default='./weihgts/potato_20220609.pth')
args = vars(parser.parse_args())

# set computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# download or load the model from disk
# model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
model = torch.load(args['model_path'], map_location=torch.device(device))
# model to eval() model and load onto computation device
model.eval().to(device)

# read the image
image = Image.open(args['input'])
# do forward pass and get the output dictionary
outputs = segmentation_utils.get_segment_labels(image, model, device)
print(f'outputs={outputs}')
# get the data from the `out` key
outputs = outputs['out']
# print(f'outputs={outputs}')
print(f'outputs max={np.max(outputs.detach().cpu().numpy())}')
check_output(outputs.squeeze(dim=0).detach().cpu().numpy())
segmented_image = segmentation_utils.draw_segmentation_map(outputs)

final_image = segmentation_utils.image_overlay(image, segmented_image)
# cv2_imshow(final_image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# show the segmented image and save to disk
# cv2.imshow('Segmented image', final_image)
# cv2.waitKey(0)
cv2.imwrite(f"./images/{save_name}_ss.jpg", final_image)
