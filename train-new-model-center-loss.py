from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from LFWDataset import LFWDataset
from logger import Logger
import argparse
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from tqdm import tqdm
from torch.autograd import Function
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test
from eval_metrics import evaluate
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser(description='PyTorch Face Recognition')
# Model options
parser.add_argument('--dataroot', type=str, default='/home/datasets/lwhu/cv/CASIA-WebFace-mtcnn-compressed-10',#default='/media/lior/LinuxHDD/datasets/vgg_face_dataset/aligned'
                    help='path to dataset')
parser.add_argument('--lfw-dir', type=str, default='/home/datasets/lwhu/cv/lfw-deepfunneled-mtcnn',
                    help='path to dataset')
parser.add_argument('--lfw-pairs-path', type=str, default='lfw_pairs.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='./models',
                    help='folder to output model checkpoints')

parser.add_argument('--resume',
                    default='./models-no-all/run-optim_adagrad-n1000000-lr0.1-wd0.0-m0.5-embeddings256-msceleb-alpha10-no-all/checkpoint_1.pth',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=100, metavar='E',
                    help='number of epochs to train (default: 50)')
# Training options
# parser.add_argument('--embedding-size', type=int, default=256, metavar='ES',
#                     help='Dimensionality of the embedding')

parser.add_argument('--center_loss_weight', type=float, default=0.5, help='weight for center loss')
parser.add_argument('--alpha', type=float, default=0.5, help='learning rate of the centers')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')

parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
LOG_DIR = args.log_dir + '/run-optim_{}-lr{}-wd{}-embeddings{}-center{}-CASIA'.format(args.optimizer, args.lr, args.wd,args.embedding_size,args.center_loss_weight)
logger = Logger(LOG_DIR)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
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

#print(model)

#model.classify = True

#img_cropped = Variable(img_cropped,requires_grad=True)

#测试单张图片的特征提取
#embedding = model(input_img.unsqueeze(0))
#print(embedding)
transform_train = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ], std = [ 0.5, 0.5, 0.5 ])
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
])
train_dir = ImageFolder(args.dataroot,transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dir,
        batch_size=args.batch_size, shuffle=True, **kwargs)
transform_test = transforms.Compose([
    transforms.Resize((160, 160)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
])



#kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=args.lfw_dir,pairs_path=args.lfw_pairs_path,
                     transform=transform_test),
    batch_size=args.batch_size, shuffle=False)



l2_dist = PairwiseDistance(2)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)
def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    top1 = AverageMeter()

    for batch_idx, (data, label) in pbar:

        data_v = Variable(data.to(device))
        target_var = Variable(label)

        # compute output
        prediction = model.forward(data_v)
        prediction = model.pred

        center_loss, model.centers = model.get_center_loss(target_var, args.alpha)

        criterion = nn.CrossEntropyLoss()

        cross_entropy_loss = criterion(prediction.to(device),target_var.to(device))

        loss = args.center_loss_weight*center_loss + cross_entropy_loss

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        # logger.log_value('cross_entropy_loss', cross_entropy_loss.data[0]).step()
        logger.log_value('total_loss', loss.item()).step()

        prec = accuracy(prediction.data, label.to(device), topk=(1,))

        top1.update(prec[0].item(), data_v.size(0))

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                'Train Prec@1 {:.2f} ({:.2f})'.format(
                    epoch, batch_idx * len(data_v), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),float(top1.val), float(top1.avg)))



    logger.log_value('Train Prec@1 ',float(top1.avg))

    # do checkpointing
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'centers': model.centers},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))



def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
       
        data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
            Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    #for i in np.arange(0.9, 1.2, 0.005):
    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels, 1.1)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer
def main():
    #img = Image.open('./1.jpg')

    #img_array=np.asarray(img)
    #print (img_array.shape)

    # Get cropped and prewhitened image tensor
    #print(img_cropped.shape)
    test_display_triplet_distance = True
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))
    # instantiate model and initialize weights



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
        else:
            checkpoint = None
            print('=> no checkpoint found at {}'.format(args.resume))

    model = InceptionResnetV1(fine_turn = 1, pretrained=None, num_classes=len(train_dir.classes),device=device)
    state_dict = {}
    cached_file1="./20180408-102900-casia-webface-logits.pt"
    cached_file2="20180408-102900-casia-webface-features.pt"
    state_dict.update(torch.load(cached_file1))
    state_dict.update(torch.load(cached_file2))
    model.load_state_dict(state_dict)
    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    start = args.start_epoch
    end = start + args.epochs
    test(test_loader, model)
    display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(1))
    """for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model)
        if test_display_triplet_distance:
            display_triplet_distance_test(model,test_loader,LOG_DIR+"/test_{}".format(epoch))
    """
if __name__ == '__main__':
    main()