import numpy as np
import optim
import torch
from utils.data_utils import load_data, load_batch
from models.models import Autoencoder
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
