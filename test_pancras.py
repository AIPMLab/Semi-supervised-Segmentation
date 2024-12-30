import os
import argparse
import torch
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from test_util import test_all_casepan

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=r'D:\lyytest\SSL-contrastive-main\dataset\Pancreas\pandata', help='Name of Experiment')
parser.add_argument('--root_pathlist', type=str, default=r'D:\lyytest\SSL-contrastive-main\dataset\Pancreas',
                    help='Name of Experiment')  # todo change dataset path
parser.add_argument('--model', type=str, default="pancras_flod0_new2", help='model_name')  # todo change test model name
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/" + FLAGS.model + '/'
test_save_path = "../model/prediction/" + FLAGS.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

# 正确读取文件路径并构建完整路径
with open(os.path.join(FLAGS.root_pathlist, 'Flods', 'test0.list'), 'r') as f:
    image_list = f.readlines()

# 使用 os.path.join 以确保正确拼接路径
image_list = [os.path.join(FLAGS.root_path, item.strip()) for item in image_list]

#D:\lyytest\SSL-contrastive-main\dataset\Pancreas\Flods\test0.list'
#D:\lyytest\SSL-contrastive-main\dataset\Pancreas\pandata\PANCREAS_0001.h5
def create_model(name='vnet'):
    # Network definition
    if name == 'vnet':
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
    if name == 'resnet34':
        net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()

    return model


def test_calculate_metric(epoch_num):
    vnet = create_model(name='vnet')
    resnet = create_model(name='resnet34')

    v_save_mode_path = os.path.join(snapshot_path, 'vnet_iter_' + str(epoch_num) + '.pth')
    vnet.load_state_dict(torch.load(v_save_mode_path))
    print("init weight from {}".format(v_save_mode_path))
    vnet.eval()

    r_save_mode_path = os.path.join(snapshot_path, 'resnet_iter_' + str(epoch_num) + '.pth')
    resnet.load_state_dict(torch.load(r_save_mode_path))
    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    print("init weight from {}".format(r_save_mode_path))
    resnet.eval()

    avg_metric = test_all_casepan(vnet, resnet, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path, dataset_pan=True)

    return avg_metric


if __name__ == '__main__':
    iters = 6000
    metric = test_calculate_metric(iters)
    print('iter:', iter)
    print(metric)
