from PIL import Image
from matplotlib.pyplot import imshow
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
from torch import nn
import time


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])
preprocess2 = transforms.Compose([
   transforms.ToTensor(),
   normalize
])
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

def Inference(image,model_ft):
    t0= time.time()

    tensor = preprocess(image)

    tensor2 = preprocess(image)
    prediction_var = Variable((tensor.unsqueeze(0)).cpu(), requires_grad=True)

    #final_layer = model_ft._modules.get('layer4')
    final_layer = model_ft._modules.get('layer4')
    activated_features = SaveFeatures(final_layer)
    prediction = model_ft(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
################
    #pred_probabilities1= pred_probabilities[1:]
    pred_probabilities1 = pred_probabilities[0:1]
    pred_probabilities2= pred_probabilities1.cpu().detach().numpy()
    ############################
    activated_features.remove()
    topk(pred_probabilities, 1)
    weight_softmax_params = list(model_ft._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    # weight_softmax_params
    print('pred_probabilities2=',pred_probabilities2)
    class_idx = topk(pred_probabilities, 1)[1].int()
    print('class_idx=', class_idx)
    class_idx1 = 0   # MKS

    zeros_tensor = torch.zeros(3, 224,224)
    #print(zeros_tensor)
    if class_idx == 0:
        overlay = getCAM(activated_features.features, weight_softmax, class_idx)
    else:
        overlay = zeros_tensor


    class_idx = topk(pred_probabilities, 2)[1].int()
    # print(pred_probabilities2)
    print("time:", round(time.time()-t0,3))
    return overlay, tensor2,pred_probabilities2






def mycapture():
    import win32gui
    import win32ui
    from ctypes import windll
    from PIL import Image

    import pygetwindow as gw
    from datetime import datetime

    # https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui


    title = 'Ethiroli'

    hwnd = win32gui.FindWindow(None, title)

    if hwnd != 0:
        # Change the line below depending on whether you want the whole window
        # or just the client area.
        # left, top, right, bot = win32gui.GetClientRect(hwnd)
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area.

        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)


        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        if result == 1:
            # PrintWindow Succeeded
            path1 = "img/"
            fname = path1+datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ".png"



            im.save(fname)              # For saving
            print("file saved with name =\t", fname)

    else:
        print("Start the application '{}', It might be closed !".format(title))

        # https://www.codeproject.com/Articles/20651/Capturing-Minimized-Window-A-Kid-s-Trick



    return im


#



############################################################ Main ######################
#Infernece start from here
def ini():
    model_ft = models.resnet18(pretrained=True)
    #model_ft = models.vgg19(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model_ft = model_ft.to(device)

    PATH = 'modelResNet18Data_samples32b.pth'
    #PATH = 'model1R50.pth'
    model_ft = torch.load(PATH, map_location=torch.device('cpu'))
    model_ft.cpu()
    model_ft.eval()
    return model_ft

def my_main():
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, draw, pause
    model_ft=ini()

    # https: // stackoverflow.com / questions / 13180941 / how - to - kill - a -
    # while -loop -with-a-keystroke
    import keyboard

    plt.figure(figsize=(12, 12))


    newflag = True
    while newflag:
        # do something
        if keyboard.is_pressed("q"):
            print("q pressed, ending loop")
            newflag= False
            break



        img = mycapture()
        area = (500, 100, 1500, 640)       # following x,y

        image = img.crop(area)

        overlay, tensor2,pred_probabilities2 = Inference(image,model_ft)

        newsize = (224, 224)


        tensor = image.resize(newsize)

        plt.subplot(1, 2, 1), imshow(tensor)

        plt.subplot(1, 2, 2), imshow(tensor)
        plt.subplot(1,2,2),imshow(skimage.transform.resize(overlay[0], tensor2.shape[1:3]), alpha=0.5, cmap='jet')

        plt.pause(0.005)
        plt.draw()
        print('hello')


#my_main()