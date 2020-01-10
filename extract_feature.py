from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

import torch





img = Image.open('./1.jpg')


mtcnn = MTCNN()

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path='./2.jpg')

model = InceptionResnetV1(pretrained=None,num_classes=10575,device="cuda:0")


state_dict = {}


cached_file1="./20180408-102900-casia-webface-logits.pt"
cached_file2="20180408-102900-casia-webface-features.pt"

state_dict.update(torch.load(cached_file1))
state_dict.update(torch.load(cached_file2))



model.load_state_dict(state_dict)




#print(model)
model.eval()



#model.classify = True


input_img=img_cropped.cuda()
embedding = model(input_img.unsqueeze(0))


print(embedding)



