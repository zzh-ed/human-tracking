# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='SiamMask_DAVIS.pth', type=str, required=False,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
#parser.add_argument('--base_path', default='../../data/VOT2018/basketball', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    cap = cv2.VideoCapture(0) # 创建摄像头识别类
    if not cap.isOpened():
        # 如果没有检测到摄像头，报错
        raise Exception('Check if the camera is on.')

    while cap.isOpened():        
        catch, im = cap.read()  # 读取每一帧图片
        cv2.imshow('SiamMask', im) # 在window上显示图片
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            # Select ROI
            cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
            try:
                init_rect = cv2.selectROI('SiamMask', im, False, False)
                x, y, w, h = init_rect
            except:
                exit()
            break

    toc = 0
    f = 0
    while True:
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break
        catch, im = cap.read()
        f += 1
        toc += cv2.getTickCount() - tic
    cap.release()
    cv2.destroyAllWindows()
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))



