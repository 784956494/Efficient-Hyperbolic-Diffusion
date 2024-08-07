import numpy as np
import optim
import torch
from utils.data_utils import load_data, load_batch
from models.autoencoder import Autoencoder
from configs import parser
import matplotlib.pyplot as plt
from torchvision.utils import save_image

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if int(args.double_precision):
    torch.set_default_dtype(torch.float64)
if int(args.cuda) >= 0:
    torch.cuda.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

train_loader, test_loader = load_data(args)

model = Autoencoder(args)
checkpoint = torch.load('/mnt/c/Users/User/Documents/Hyperbolic-Embedding-Methods/checkpoints/mnist/14.pth')
model.load_state_dict(checkpoint['AE_state'])

i = 0
for image, _ in test_loader:
    print(i)
    image = load_batch(args, image)
    item = image.reshape(-1, 28, 28)
    save_image(item[0].cpu(), 'img'+str(i)+'.png')
    reconstructed = (image)
    item = reconstructed.reshape(-1, 28, 28)
    save_image(item[0].cpu(), 'rec_img'+str(i)+'.png')
    i+=1