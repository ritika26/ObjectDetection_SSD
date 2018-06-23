import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

def detect(img,net,transform):
    height,width= img.shape[:2]
    # # We apply the transformation to our original image
    img_t = transform(img)[0] 
    # We convert the image into a torch tensor.
    x = torch.from_numpy(img_t).permute(2, 0, 1) 
    # We add a fake dimension corresponding to the batch.
    x = Variable(x.unsqueeze(0))
    # We feed the neural network ssd with the image and we get the output y.
    y = net(x)
    # We create the detections tensor contained in the output y.
    detections = y.data 
    # We create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height]) 
    for i in range(detections.size(1)): # For every class:
        j = 0 
        while detections[0, i, j, 0] >= 0.6: 
            pt = (detections[0, i, j, 1:] * scale).numpy() 
            cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 
            cv2.putText(img, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
            j += 1 # We increment j to get to the next occurrence.
    return img # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.

# Doing some Object Detection on a video

# We open the video.
reader = imageio.get_reader('finndog.mp4') 

# We get the fps frequence (frames per second).
fps = reader.get_meta_data()['fps'] 
# We create an output video with this same fps frequence.
writer = imageio.get_writer('output.mp4', fps = fps) 

for i, img in enumerate(reader): # We iterate on the frames of the output video:
    img = detect(img, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(img) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.


