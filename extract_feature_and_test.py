from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import torch.nn as nn
from LFWDataset import LFWDataset
import argparse
from torch.autograd import Variable
from tqdm import tqdm
from torch.autograd import Function
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test
from eval_metrics import evaluate
import torchvision.transforms as transforms
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class enhanceNet(nn.Module):
    def __init__(self):
        super(enhanceNet, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, padding_mode='same'),
            nn.LeakyReLU(0.2),
            resuial_block(32, 64),
            resuial_block(64, 64),
            resuial_block(64, 64),
            nn.Conv2d(64, 32, 3, padding=1, padding_mode='same'),
            nn.LeakyReLU(0.2))
        self.last = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        y = x
        x = self.base(x)
        x = self.last(x)
        return x + y
class resuial_block(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, ch_in, ch_out):
        super(resuial_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, 64, 3, padding=1, padding_mode='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, ch_out, 3, padding=1, padding_mode='same'),
        )
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1, padding=0, padding_mode='same')
    def forward(self, x):
        y = self.lrelu(self.conv(x)) + self.conv1x1(x)
        return y
parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
# Model options

parser.add_argument('--device', type=str, default='cuda',
                    help='path to pairs file')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='input batch size for training (default: 128)')


parser.add_argument('--resume',
                    default='/home/lwhu/cv_project/models-test/hhh/checkpoint_49.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lfw-dir', type=str, default='/home/datasets/lwhu/cv/lfw-deepfunneled-mtcnn',
                    help='path to dataset')

parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')

parser.add_argument('--enhanceNet-dir', type=str, default='/home/datasets/lwhu/cv/modles-mtcnn-mix/enhanceNet_49',
                    help='path to enhanceNet file')

parser.add_argument('--enhanceNet-flag', type=int, default=1,
                    help='path to enhanceNet file')

args = parser.parse_args()
device1=args.device

img = Image.open('./1.jpg')

#img_array=np.asarray(img)
#print (img_array.shape)

mtcnn = MTCNN()

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path='./2.jpg')


#print(img_cropped.shape)


if args.enhanceNet_flag == 1:
    ennet = enhanceNet()
    ennet = ennet.cuda()
    ck = torch.load("/home/datasets/lwhu/cv/modles-mtcnn-mix/enhanceNet_49")
    ennet.load_state_dict(ck)
    ennet.eval()
else:
	ennet = None

model = InceptionResnetV1(pretrained=None,num_classes=10575,device=device1)
checkpoint = torch.load("/home/lwhu/cv_project/facenet-sunhongbo/models-enhance/run-optim_adam-lr0.001-wd0.0-embeddings512-center0.5-CASIA/checkpoint_5.pth")
state = checkpoint["state_dict"]
model.load_state_dict(state)
'''
state_dict = {}
cached_file1="./20180408-102900-casia-webface-logits.pt"
cached_file2="20180408-102900-casia-webface-features.pt"
state_dict.update(torch.load(cached_file1))
state_dict.update(torch.load(cached_file2))
model.load_state_dict(state_dict)
'''
	

def enhance_image(data_tensor, ennet):
    n, c, h, w = data_tensor.size()
    data_numpy = data_tensor.cpu().numpy()
    dct_numpy = np.zeros_like(data_numpy)
    # dct
    for batch in range(n):
        for channel in range(c):
            for i in range(0, h, 10):
                for j in range(0, w, 10):
                    dct_numpy[batch, channel, i: i + 10, j: j + 10] = cv2.dct(
                        data_numpy[batch, channel, i: i + 10, j: j + 10])
    dct_tensor = torch.from_numpy(dct_numpy)
    dct_tensor = dct_tensor.cuda()
    enhance_dct_tensor = ennet(dct_tensor)
    enhance_dct_numpy = enhance_dct_tensor.cpu().detach().numpy()
    ret_numpy = np.zeros_like(data_numpy)
    # idct
    
    for batch in range(n):
        for channel in range(c):
            for i in range(0, h, 10):
                for j in range(0, w, 10):
                    ret_numpy[batch, channel, i: i + 10, j: j + 10] = cv2.idct(
                        enhance_dct_numpy[batch, channel, i: i + 10, j: j + 10])
    ret_tensor = torch.from_numpy(ret_numpy)
    return ret_tensor.cuda()
#print(type(state_dict))

#print(model)
model.eval()

#model.classify = True
input_img=img_cropped.cuda()

#img_cropped = Variable(img_cropped,requires_grad=True)

#测试单张图片的特征提取
#embedding = model(input_img.unsqueeze(0))
#print(embedding)




transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])



#kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                     transform=transform_test),
    batch_size=args.batch_size, shuffle=False)



l2_dist = PairwiseDistance(2)






def test(test_loader, model, ennet):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
       
        data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
            Variable(data_p, volatile=True), Variable(label)

        # compute output
        if args.enhanceNet_flag == 1:
            data_a = enhance_image(data_a, ennet)
            data_p = enhance_image(data_p, ennet)
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])


    mx_ac = 0
    floor = 0
    #for i in np.arange(0.8, 1.4, 0.005):
    tpr, fpr, accuracy = evaluate(distances,labels, 1.16)
    temp = np.mean(accuracy)
    if temp >= mx_ac:
    	mx_ac = temp
    	print(mx_ac)
    print("12345678910")
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(mx_ac))
    print(floor)

test(test_loader, model, ennet)