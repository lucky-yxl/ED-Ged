import os
import torch
import numpy as np

from PIL import Image
import torch.nn as nn
from torch.utils import data

from network import *
from dataset.zurich_night_dataset import zurich_night_DataSet, nightcity_DataSet
from configs.test_config import get_arguments

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def main():
    args = get_arguments()

    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda")
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes, dgf=args.DGF_FLAG)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, dgf=args.DGF_FLAG)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False, dgf=args.DGF_FLAG)

    saved_state_dict = torch.load(args.restore_from)

    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    model.load_state_dict(saved_state_dict)

    model = model.to(device)
    model.eval()

    if args.NIGHTCITY_FLAG:
        testloader = data.DataLoader(nightcity_DataSet(args.data_dir, args.data_list, set=args.set))
        interp = nn.Upsample(size=(512,1024), mode='bilinear', align_corners=True)
    elif args.CITYSCAPES_FLAG:
        testloader = data.DataLoader(nightcity_DataSet(args.data_dir, args.data_list, set=args.set))
        interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    else:
        testloader = data.DataLoader(zurich_night_DataSet(args.data_dir, args.data_list, set=args.set))
        interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    for index, batch in enumerate(testloader):
        if index % 50 == 0:
            print('%d processed' % index)
        image, name = batch
        image = image.to(device)
        name = name[0].split('/')[-1]

        with torch.no_grad():
            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                output2 = model(image)
            else:
                _, output2 = model(image)

        output = interp(output2).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        output.save('%s/%s' % (args.save, name))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))


if __name__ == '__main__':
    main()
