## Code based is based on https://github.com/WYC-321/MCF paper.
import os
import sys

from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.vnet import VNet
from networks.ResNet34 import Resnet34
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default=r"D:\lyytest\SSL-contrastive-main\dataset\LA\LAdata", help='Name of Experiment')
parser.add_argument('--root_pathlist', type=str, default=r'D:\lyytest\SSL-contrastive-main\dataset\LA',
                    help='Name of Experiment')  # todo change dataset path
parser.add_argument('--exp', type=str, default="MCF_flod0", help='model_name')  # todo model name
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
train_data_pathlist = args.root_pathlist

snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
T = 0.1
Good_student = 0  # 0: vnet 1:resnet


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def gateher_two_patch(vec):
    b, c, num = vec.shape
    cat_result = []
    for i in range(c - 1):
        temp_line = vec[:, i, :].unsqueeze(1)  # b 1 c
        star_index = i + 1
        rep_num = c - star_index
        repeat_line = temp_line.repeat(1, rep_num, 1)
        two_patch = vec[:, star_index:, :]
        temp_cat = torch.cat((repeat_line, two_patch), dim=2)
        cat_result.append(temp_cat)

    result = torch.cat(cat_result, dim=1)
    return result


def uncertainty_rectified_loss(inputs, targets, T=0.1):
    """
    Uncertainty rectified pseudo supervised loss with temperature scaling.
    """
    # Step 1: Compute softmax for teacher logits (targets) with temperature scaling
    pseudo_label = F.softmax(targets / T, dim=1).detach()

    # Convert pseudo_label to integer class indices by taking the class with max probability
    pseudo_label = torch.argmax(pseudo_label, dim=1)  # Convert to class indices (Long type)

    # Step 2: Compute the vanilla cross-entropy loss between student inputs and pseudo-label
    vanilla_loss = F.cross_entropy(inputs, pseudo_label, reduction='none')  # pseudo_label should be class indices

    # Step 3: Compute KL divergence between student (inputs) and teacher (targets)
    kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)

    # Step 4: Compute uncertainty weight based on KL divergence
    uncertainty_weight = torch.exp(-kl_div)

    # Step 5: Compute the uncertainty rectified loss
    uncertainty_loss = (uncertainty_weight * vanilla_loss).mean() + kl_div.mean()

    return uncertainty_loss





def loss_fn(candidates, prototype):
    x = F.normalize(candidates, dim=0, p=2).permute(1, 0).unsqueeze(0)
    y = F.normalize(prototype, dim=0, p=2).permute(1, 0).unsqueeze(0)

    loss = torch.cdist(x, y, p=2.0).mean()

    return loss

def compute_uxi_loss_with_mean_teacher(predict_student_a, predict_student_b, predict_teacher_a, predict_teacher_b, represent_student_a, percent=20):
    """
    基于 Mean Teacher 的伪标签生成方法。
    predict_student_a, predict_student_b: 学生模型的预测输出
    predict_teacher_a, predict_teacher_b: 教师模型的预测输出
    represent_student_a: 学生模型的特征表示
    """
    batch_size, num_class, h, w, d = predict_student_a.shape

    # 学生模型预测
    logits_student_a, label_student_a = torch.max(predict_student_a, dim=1)
    logits_student_b, label_student_b = torch.max(predict_student_b, dim=1)

    # 教师模型预测
    logits_teacher_a, label_teacher_a = torch.max(predict_teacher_a, dim=1)
    logits_teacher_b, label_teacher_b = torch.max(predict_teacher_b, dim=1)

    # 使用教师模型的预测作为伪标签生成的依据
    target = label_teacher_a | label_teacher_b

    with torch.no_grad():
        # 从教师模型中过滤掉高不确定性的像素（a）
        prob_teacher_a = predict_teacher_a
        entropy_teacher_a = -torch.sum(prob_teacher_a * torch.log(prob_teacher_a + 1e-10), dim=1)

        thresh_teacher_a = np.percentile(entropy_teacher_a.detach().cpu().numpy().flatten(), percent)
        thresh_mask_teacher_a = entropy_teacher_a.ge(thresh_teacher_a).bool()

        # 从教师模型中过滤掉高不确定性的像素（b）
        prob_teacher_b = predict_teacher_b
        entropy_teacher_b = -torch.sum(prob_teacher_b * torch.log(prob_teacher_b + 1e-10), dim=1)

        thresh_teacher_b = np.percentile(entropy_teacher_b.detach().cpu().numpy().flatten(), percent)
        thresh_mask_teacher_b = entropy_teacher_b.ge(thresh_teacher_b).bool()

        # 只保留教师模型在 a 和 b 中的低不确定性像素
        thresh_mask = torch.logical_and(thresh_mask_teacher_a, thresh_mask_teacher_b)

        # 对高不确定性的像素进行伪标签标注
        target[thresh_mask] = 2

        # 根据 target 计算特征表示
        target_clone = torch.clone(target.view(-1))
        represent_student_a = represent_student_a.permute(1, 0, 2, 3, 4)
        represent_student_a = represent_student_a.contiguous().view(represent_student_a.size(0), -1)

        # 分离前景和背景的原型
        prototype_f = represent_student_a[:, target_clone == 1].mean(dim=1)
        prototype_b = represent_student_a[:, target_clone == 0].mean(dim=1)

        # 从未标注像素中挑选前景和背景候选样本
        foreground_candidate = represent_student_a[:, (target_clone == 2) & (label_student_a.view(-1) == 1)]
        background_candidate = represent_student_a[:, (target_clone == 2) & (label_student_a.view(-1) == 0)]

        num_samples = foreground_candidate.size(1) // 100
        selected_indices_f = torch.randperm(foreground_candidate.size(1))[:num_samples]
        selected_indices_b = torch.randperm(background_candidate.size(1))[:num_samples]

        # 计算对比损失
        contrastive_loss_f = loss_fn(foreground_candidate[:, selected_indices_f], prototype_f.unsqueeze(dim=1))
        contrastive_loss_b = loss_fn(background_candidate[:, selected_indices_b], prototype_b.unsqueeze(dim=1))
        contrastive_loss_c = loss_fn(prototype_f.unsqueeze(dim=1), prototype_b.unsqueeze(dim=1))

        # 基于学生模型和教师模型特征的对比损失
        # contrastive_loss_fn_value = contrastive_loss_fnori(represent_student_a, represent_student_a)

        # 总的对比损失
        con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c
                   # + contrastive_loss_fn_value

        # 根据伪标签权重计算损失
        weight = batch_size * h * w * d / torch.sum(target != 2)

    # 学生模型的交叉熵损失
    loss_student_a = weight * F.cross_entropy(predict_student_a, target, ignore_index=2)
    loss_student_b = weight * F.cross_entropy(predict_student_b, target, ignore_index=2)

    return loss_student_a, loss_student_b, con_loss

def get_confidence_prediction(prediction, percent=40):
    _, target = torch.max(prediction, dim=1)
    with torch.no_grad():
        prob = prediction
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        thresh = np.percentile(
            entropy.detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool()
        target[thresh_mask] = 0
    return target
class GumbelTopK(nn.Module):
    """
    Perform top-k or Gumble top-k on given data
    """
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        """
        Args:
            k: int, number of chosen
            dim: the dimension to perform top-k
            gumble: bool, whether to introduce Gumble noise when sampling
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        # logits shape: [B, N], B denotes batch size, and N denotes the multiplication of channel and spatial dim
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)
#
#
class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        # 确保 feat_q 和 feat_k 的形状匹配
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())

        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]

        # Reshape 和 permute，转换成合适的形状
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)

        # 归一化
        feat_q = F.normalize(feat_q, dim=-1, p=2)
        feat_k = F.normalize(feat_k, dim=-1, p=2)
        feat_k = feat_k.detach()  # 不计算 feat_k 的梯度

        # 正样本对计算
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  # (batch_size * num_patches, 1, dim)
        l_pos = l_pos.view(-1, 1)  # 展平为 (batch_size * num_patches, 1)

        # 负样本对计算
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1  # 所有样本看作同一个批次处理
        else:
            batch_dim_for_bmm = batch_size  # 每个批次独立处理

        # reshape features for batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)

        npatches = feat_q.size(1)  # patch 数量
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # 负样本对 (batch_size, num_patches, num_patches)

        # 创建对角掩码，避免相同样本比较
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))  # 将对角线上的值设为无穷小
        l_neg = l_neg_curbatch.view(-1, npatches)  # (batch_size * num_patches, num_patches)

        # 拼接正负样本对并除以温度参数
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        # 计算对比损失，目标为第一个位置为正样本
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

        return loss
contrastive_loss_fn = contrastive_loss_sup(temperature=0.07, base_temperature=0.07)

def compute_uxi_loss(predicta, predictb, represent_a, represent_b, percent=20):
    batch_size, num_class, h, w, d = predicta.shape

    logits_u_a, label_u_a = torch.max(predicta, dim=1)
    logits_u_a, label_u_b = torch.max(predictb, dim=1)

    target = label_u_a | label_u_b
    with torch.no_grad():
        # drop pixels with high entropy from a
        prob_a = predicta

        entropy_a = -torch.sum(prob_a * torch.log(prob_a + 1e-10), dim=1)

        thresh_a = np.percentile(
            entropy_a.detach().cpu().numpy().flatten(), percent
        )

        thresh_mask_a = entropy_a.ge(thresh_a).bool()

        # drop pixels with high entropy from b
        prob_b = predictb

        entropy_b = -torch.sum(prob_b * torch.log(prob_b + 1e-10), dim=1)

        thresh_b = np.percentile(
            entropy_b.detach().cpu().numpy().flatten(), percent
        )

        thresh_mask_b = entropy_b.ge(thresh_b).bool()

        thresh_mask = torch.logical_and(thresh_mask_a, thresh_mask_b)

        target[thresh_mask] = 2

        target_clone = torch.clone(target.view(-1))
        represent_a = represent_a.permute(1, 0, 2, 3, 4)
        # print(represent_a.size())
        represent_a = represent_a.contiguous().view(represent_a.size(0), -1)
        prototype_f = represent_a[:, target_clone == 1].mean(dim=1)
        prototype_b = represent_a[:, target_clone == 0].mean(dim=1)

        forground_candidate = represent_a[:, (target_clone == 2) & (label_u_a.view(-1) == 1)]
        background_candidate = represent_a[:, (target_clone == 2) & (label_u_a.view(-1) == 0)]

        num_samples = forground_candidate.size(1) // 100
        selected_indices_f = torch.randperm(forground_candidate.size(1))[:num_samples]
        selected_indices_b = torch.randperm(background_candidate.size(1))[:num_samples]

        contrastive_loss_f = loss_fn(forground_candidate[:, selected_indices_f], prototype_f.unsqueeze(dim=1))
        contrastive_loss_b = loss_fn(background_candidate[:, selected_indices_b], prototype_b.unsqueeze(dim=1))
        contrastive_loss_c = loss_fn(prototype_f.unsqueeze(dim=1), prototype_b.unsqueeze(dim=1))
        print(represent_a.shape)
        print(represent_b.shape)

        contrastive_loss_fn_value = contrastive_loss_fn(represent_a, represent_a)
        # contrastive_loss_fn_value = contrastive_loss_fn(represent_b, represent_b)
        # contrastive_loss_fn_value = 0.5*contrastive_loss_fn_value1 + 0.5* contrastive_loss_fn_value2
        con_loss = contrastive_loss_f + contrastive_loss_b + contrastive_loss_c + contrastive_loss_fn_value

        weight = batch_size * h * w * d / torch.sum(target != 2)

    loss_a = weight * F.cross_entropy(predicta, target, ignore_index=2)
    loss_b = weight * F.cross_entropy(predictb, target, ignore_index=2)

    return loss_a, loss_b, con_loss

def consist_loss(inputs, targets):
    """
    Consistency regularization between two augmented views
    """
    # Compute the cosine similarity between the two tensors
    loss = (1.0 - F.cosine_similarity(inputs, targets, dim=1)).mean()
    return loss


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(name='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model


    model_vnet = create_model(name='vnet')
    model_resnet = create_model(name='resnet34')

    db_train = LAHeart(base_dir=train_data_path, base_dir_list=train_data_pathlist,
                       split='train',
                       train_flod='train0.list',  # todo change training flod
                       common_transform=transforms.Compose([
                           RandomCrop(patch_size),
                       ]),
                       sp_transform=transforms.Compose([
                           ToTensor(),
                       ]))

    labeled_idxs = list(range(16))  # todo set labeled num
    unlabeled_idxs = list(range(16, 80))  # todo set labeled num all_sample_num

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    vnet_optimizer = optim.SGD(model_vnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    resnet_optimizer = optim.SGD(model_resnet.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    model_vnet.train()
    model_resnet.train()
    Thresh = 0.6
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('epoch:{},i_batch:{}'.format(epoch_num, i_batch))
            volume_batch1, volume_label1 = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2, volume_label2 = sampled_batch[1]['image'], sampled_batch[1]['label']

            v_input, v_label = volume_batch1.cuda(), volume_label1.cuda()
            r_input, r_label = volume_batch2.cuda(), volume_label2.cuda()

            v_outputs, v_rep = model_vnet(v_input)
            r_outputs, r_rep = model_resnet(r_input)


            lurp_loss = uncertainty_rectified_loss(r_outputs, v_outputs, T=0.5)  # You can adjust T as needed输出应该是同样的
            # lurp_loss = torch.clamp(lurp_loss, max=1e4)
            # Now, calculate the contrastive loss using the features v_rep and r_rep
            # contrastive_loss_value = contrastive_loss_fn(v_rep, r_rep)  # Call your contrastive loss
            # # contrastive_loss_value2 = contrastive_loss_fn(r_rep, r_rep)
            # # contrastive_loss_value = 0.5*contrastive_loss_value1 + 0.5*contrastive_loss_value2
            ## calculate the supervised loss
            v_loss_seg = F.cross_entropy(v_outputs[:labeled_bs], v_label[:labeled_bs])
            v_outputs_soft = F.softmax(v_outputs, dim=1)
            v_loss_seg_dice = losses.dice_loss(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs] == 1)

            r_loss_seg = F.cross_entropy(r_outputs[:labeled_bs], r_label[:labeled_bs])
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_loss_seg_dice = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs] == 1)

            # Calculate the consistency loss between the outputs from the two views
            consistency_loss_value = consist_loss(v_outputs_soft, r_outputs_soft)#输出应该是一致的
            if v_loss_seg_dice < r_loss_seg_dice:
                winner = 0
            else:
                winner = 1

            ## Cross reliable loss term
            v_probability, v_predict = torch.max(v_outputs_soft[:labeled_bs, :, :, :, :], 1, )
            r_probability, r_predict = torch.max(r_outputs_soft[:labeled_bs, :, :, :, :], 1, )
            conf_diff_mask = (((v_predict == 1) & (v_probability >= Thresh)) ^ (
                        (r_predict == 1) & (r_probability >= Thresh))).to(torch.int32)

            v_mse_dist = consistency_criterion(v_outputs_soft[:labeled_bs, 1, :, :, :], v_label[:labeled_bs])
            r_mse_dist = consistency_criterion(r_outputs_soft[:labeled_bs, 1, :, :, :], r_label[:labeled_bs])
            v_mistake = torch.sum(conf_diff_mask * v_mse_dist) / (torch.sum(conf_diff_mask) + 1e-16)
            r_mistake = torch.sum(conf_diff_mask * r_mse_dist) / (torch.sum(conf_diff_mask) + 1e-16)

            v_supervised_loss = (v_loss_seg + v_loss_seg_dice) + 0.5 * v_mistake
            r_supervised_loss = (r_loss_seg + r_loss_seg_dice) + 0.5 * r_mistake

            v_outputs_clone = v_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = r_outputs_soft[labeled_bs:, :, :, :, :].clone().detach()

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            if winner == 0:
                loss_u_r, loss_u_v, con_loss = compute_uxi_loss_with_mean_teacher(
                    r_outputs_clone,  # 学生模型预测
                    v_outputs_clone,  # 学生模型预测
                    r_outputs[labeled_bs:],  # 教师模型预测
                    v_outputs[labeled_bs:],  # 教师模型预测
                    r_rep[labeled_bs:],  # 学生模型的特征表示
                    percent=20
                )

                v_loss = v_supervised_loss + loss_u_v
                r_loss = r_supervised_loss + loss_u_r + consistency_weight * con_loss +lurp_loss+consistency_loss_value
            else:
                loss_u_v, loss_u_r, con_loss = compute_uxi_loss_with_mean_teacher(
                    v_outputs_clone,  # 学生模型预测
                    r_outputs_clone,  # 学生模型预测
                    v_outputs[labeled_bs:],  # 教师模型预测
                    r_outputs[labeled_bs:],  # 教师模型预测
                    v_rep[labeled_bs:],  # 学生模型的特征表示
                    percent=20
                )

                v_loss = v_supervised_loss + loss_u_v + consistency_weight * con_loss+lurp_loss+consistency_loss_value
                r_loss = r_supervised_loss + loss_u_r
            vnet_optimizer.zero_grad()
            resnet_optimizer.zero_grad()
            v_loss.backward(retain_graph=True)
            r_loss.backward()
            vnet_optimizer.step()
            resnet_optimizer.step()
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/ consistency_loss_value', consistency_loss_value, iter_num)
            writer.add_scalar('loss/ lurp_loss',  lurp_loss,  iter_num)
            writer.add_scalar('loss/v_loss', v_loss, iter_num)
            writer.add_scalar('loss/v_loss_seg', v_loss_seg, iter_num)
            writer.add_scalar('loss/v_loss_seg_dice', v_loss_seg_dice, iter_num)
            writer.add_scalar('loss/v_supervised_loss', v_supervised_loss, iter_num)
            writer.add_scalar('loss/con_loss', con_loss, iter_num)
            writer.add_scalar('loss/r_loss', r_loss, iter_num)
            writer.add_scalar('loss/r_loss_seg', r_loss_seg, iter_num)
            writer.add_scalar('loss/r_loss_seg_dice', r_loss_seg_dice, iter_num)
            writer.add_scalar('loss/r_supervised_loss', r_supervised_loss, iter_num)
            writer.add_scalar('train/Good_student', Good_student, iter_num)

            logging.info(
                'iteration ： %d   consistency_loss_value:%f lurp_loss:%f v_supervised_loss : %f v_loss_seg : %f v_loss_seg_dice : %f r_supervised_loss : %f r_loss_seg : %f r_loss_seg_dice : %f Good_student: %f' %
                (iter_num,   consistency_loss_value.item(), lurp_loss.item(),
                 v_supervised_loss.item(), v_loss_seg.item(), v_loss_seg_dice.item(),
                 r_supervised_loss.item(), r_loss_seg.item(), r_loss_seg_dice.item(), Good_student))

            ## change lr

            if iter_num % 2500 == 0 and iter_num != 0:
                lr_ = lr_ * 0.1
                for param_group in vnet_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in resnet_optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num >= max_iterations:
                break
            time1 = time.time()

            iter_num = iter_num + 1
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    save_mode_path_vnet = os.path.join(snapshot_path, 'vnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_vnet.state_dict(), save_mode_path_vnet)
    logging.info("save model to {}".format(save_mode_path_vnet))

    save_mode_path_resnet = os.path.join(snapshot_path, 'resnet_iter_' + str(max_iterations) + '.pth')
    torch.save(model_resnet.state_dict(), save_mode_path_resnet)
    logging.info("save model to {}".format(save_mode_path_resnet))

    writer.close()
