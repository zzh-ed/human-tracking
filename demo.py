import glob
import os
from tools.inference_function import *
import shutil
import time


first_reID_thres=0.5#0.5
later_reID_thres=0.75
track_thres=0.92
update_query_thres=0.999









parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='./cfg/SiamMask_DAVIS.pth', type=str, required=False,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='./cfg/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--file_index', default=0, type=int, required=False)

parser.add_argument('--query_img', default='', type=str, required=True,
                    metavar='path to query images',help='path to query images (default: none)')
parser.add_argument('--video', default='', type=str, required=True,
                    metavar='path to video',help='path to video (default: none)')

args = parser.parse_args()




if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    # Setup Model
    cfg = load_config(args)
    from tools.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)


    videoinpath = args.video
    video_name= videoinpath.split("/")[-1]
    videooutpath = './output_video/output_%s'%video_name
    first_init=True
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is unavailable')

    print("video name: ",videoinpath.split("/")[-1])
    cap    = cv2.VideoCapture(videoinpath)
    fourcc      = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    shutil.rmtree('./match_history')
    os.mkdir('./match_history')
    shutil.rmtree('./roi_lib/query/img')
    os.mkdir('./roi_lib/query/img')
    shutil.copy(args.query_img,"./roi_lib/query/img/1.jpg")
    #帧率
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # 分辨率-宽度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 分辨率-高度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("fps : ",fps)
    print("width : ",width)
    print("height : ",height)
    writer      = cv2.VideoWriter(videooutpath ,fourcc, fps, (width,height), True)
    Flag=False
    if cap.isOpened():
        print('-------Tracking-----------')
        while True:
            ret,im=cap.read()
            if not ret:break
            if Flag: #tracking
                state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
                print("Tracking score : ",state['score'])
                if state['score']>track_thres:
                    if state['score']>update_query_thres:
                        w, h = int(state['target_sz'][0]),int(state['target_sz'][1])
                        x, y = int(state['target_pos'][0]-w/2), int(state['target_pos'][1]-h/2)                  
                        ROI=im[y:y+h,x:x+w]
                        cv2.imwrite('./roi_lib/query/img/1.jpg', ROI)
                        print("Update the query image")

                    location = state['ploygon'].flatten()
                    mask = state['mask'] > state['p'].seg_thr
                    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
                    cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)

                else:
                    Flag=False
                
            if not Flag: # init
                #try:
                #    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
                #    init_rect = cv2.selectROI('SiamMask', im, False, False)
                #    x, y, w, h = x, y, w, h = init_rect
                #    score=1
                #except:
                #    exit()
                print("Looking for the target...")                
                start = time.time()
                x, y, w, h, score = yolo_detect(im)
                if first_init:  
                   reID_thres=first_reID_thres
                   first_init=False
                else: reID_thres= later_reID_thres
                if score > reID_thres :                   
                    target_pos = np.array([x + w / 2, y + h / 2])
                    target_sz = np.array([w, h])
                    state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
                    Flag=True
                end = time.time()
                print('Takes {:.2f} s . '.format(end - start))
                
                       
            writer.write(im)
    else:
        print('The video fail to be open')

    print(('----------Done-----------'))

