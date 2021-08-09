import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from generate_data.gt_utils import get_object_frame

from IPython import embed

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
gt_path = '/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/track_dataset/gt.txt'

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

# def draw_gt_boxes(img, bbox, identities=None, offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img

def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')

    # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
    #                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    #                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    #                     use_cuda=True, reid_classes=cfg.DEEPSORT.REID_CLASSES_DIM)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    # save_path = 'inference/test_output/'
    # txt_path = os.path.join(save_path, 'test_pred')


    group_frame = get_object_frame(gt_path)

    '''
    =================================================================================
    ''' 
    # # contain extracted frames
    # frame_dir = ''

    # # contain list of txt pred file coresponding for each frame
    # pred_dir = ''
    
    # for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)

    #     # Inference
    #     t1 = time_synchronized()
    #     pred = model(img, augment=opt.augment)[0]

    #     embed(header='debug pred')

    #     # Apply NMS

    #     # Returns: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    #     pred = non_max_suppression(
    #         pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #     t2 = time_synchronized()

    #     embed(header='debug pred after nms')


        
    #     # Example: [tensor([[444.50000,  33.65625, 492.00000, 113.87500,   0.76025,  58.00000]], device='cuda:0')]

    #     # img0 shape: 1080 x 1920 x 3
    #     # s: 384x640

    #     # Process detections
    #     for i, det in enumerate(pred):  # detections per image

    #         # det example: tensor([[444.50000,  33.65625, 492.00000, 113.87500,   0.76025,  58.00000]], device='cuda:0')
    #         if webcam:  # batch_size >= 1
    #             p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    #         else:
    #             '''
    #             p: path to vid
    #             s: size (384 x 640)
    #             im0: ndarray (1080 x 1920 x 3)
    #             '''
    #             p, s, im0 = path, '', im0s

    #         s += '%gx%g ' % img.shape[2:]  # print string  '384x640'

    #         # 'inference/output/NVR-CH01_S20210608-084648_E20210608-084709.mp4'
    #         save_path = str(Path(out) / Path(p).name)
            
    #         embed(header='debug det')

    #         if det is not None and len(det):
    #             # Rescale boxes from img_size to im0 size (from 384x640 to 1080x1920x3)
    #             det[:, :4] = scale_coords(
    #                 img.shape[2:], det[:, :4], im0.shape).round()

    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

    #             xywh_bboxs = []
    #             confs = []

    #             embed(header = 'debug scale box')
    #             # Adapt detections to deep sort input format
    #             for *xyxy, conf, cls in det:
    #                 '''
    #                 *xyxy: tensor(1334., device='cuda:0') tensor(65., device='cuda:0') tensor(1476., device='cuda:0') tensor(306., device='cuda:0')
    #                 conf: tensor(0.76025, device='cuda:0')
    #                 cls: tensor(58., device='cuda:0')
    #                 '''
    #                 # to deep sort format
    #                 x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy) # convert to x center, y center, w, h
    #                 xywh_obj = [x_c, y_c, bbox_w, bbox_h]
    #                 xywh_bboxs.append(xywh_obj)
    #                 confs.append([conf.item()])

    #             xywhs = torch.Tensor(xywh_bboxs)    # tensor([[1405.00000,  185.50000,  142.00000,  241.00000]]) (1 x 4)
    #             confss = torch.Tensor(confs)        # tensor([[0.76025]]) (1 x 1)
                
    #             # pass detections to deepsort

    #             # update tracker, return tensor of xy_xy + track id 
    #             outputs = deepsort.update(xywhs, confss, im0)

    #             embed(header='debug input for deepsort')

    #             # draw boxes for visualization
    #             if len(outputs) > 0 and opt.mode == "predict":
    #                 bbox_xyxy = outputs[:, :4]
    #                 identities = outputs[:, -1]
    #                 draw_boxes(im0, bbox_xyxy, identities)
    #                 # to MOT format
    #                 tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

    #                 # Write MOT compliant results to file
    #                 if save_txt:
    #                     for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
    #                         bbox_top = tlwh_bbox[0]
    #                         bbox_left = tlwh_bbox[1]
    #                         bbox_w = tlwh_bbox[2]
    #                         bbox_h = tlwh_bbox[3]
    #                         identity = output[-1]
    #                         with open(txt_path, 'a') as f:
    #                             f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
    #                                                         bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
    #             if(opt.mode == "gt"):
    #                 persons_inf = group_frame[frame_idx+1]
    #                 bbox_gt_xyxy = list(map(lambda x: x.xywh_to_xyxy(), persons_inf))
    #                 identities_gt = list(map(lambda x: x.track_id, persons_inf))
    #                 draw_boxes(im0, bbox_gt_xyxy, identities_gt)

    #         else:
    #             deepsort.increment_ages()

    #         # Print time (inference + NMS)
    #         print('%sDone. (%.3fs)' % (s, t2 - t1))

    #         # Stream results
    #         if show_vid:
    #             cv2.imshow(p, im0)
    #             if cv2.waitKey(1) == ord('q'):  # q to quit
    #                 raise StopIteration

    #         # Save results (image with detections)
    #         if save_vid:
    #             if vid_path != save_path:  # new video
    #                 vid_path = save_path
    #                 if isinstance(vid_writer, cv2.VideoWriter):
    #                     vid_writer.release()  # release previous video writer
    #                 if vid_cap:  # video
    #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                 else:  # stream
    #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                     save_path += '.mp4'

    #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #             vid_writer.write(im0)

    

    # contain extracted frames
    frame_dir = opt.frame_dir

    # contain list of txt pred file coresponding for each frame
    pred_dir = opt.det_pred_dir

    print(frame_dir)
    print(pred_dir)


    # for each frame
    for file in sorted(os.listdir(pred_dir)):
        frame_idx = int(file[6:12])

        im0 = cv2.imread(os.path.join(frame_dir, file[:-3] + 'jpg'))
        
        f = open(os.path.join(pred_dir, file), 'r')
        lines = f.readlines()

        xywh_bboxs = []
        confs = []

        # for each det in the current images
        for line in lines:
            line = line.strip()

            # class conf left top right bottom 
            det = (line.split(' '))

            det[0] = 0
            for i in range(1, 6):
                det[i] = float(det[i])

            det = torch.Tensor([det])
            det.to(device)

            if det is not None and len(det):
                for cls, conf, *xyxy in det:
                    '''
                    *xyxy: tensor(1334., device='cuda:0') tensor(65., device='cuda:0') tensor(1476., device='cuda:0') tensor(306., device='cuda:0')
                    conf: tensor(0.76025, device='cuda:0')
                    cls: tensor(58., device='cuda:0')
                    '''
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy) # convert to x center, y center, w, h
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)    # tensor([[1405.00000,  185.50000,  142.00000,  241.00000]]) (1 x 4)
                confss = torch.Tensor(confs)        # tensor([[0.76025]]) (1 x 1)
        
                # pass detections to deepsort

                # update tracker, return tensor of xy_xy + track id 
                outputs = deepsort.update(xywhs, confss, im0)


                # embed()

                if len(outputs) > 0 and opt.mode == "predict":
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # embed()
                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                # f.write(('%g ' * 10 + '\n') % (frame_idx,  identity, bbox_top,
                                #                                     bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                                f.write('{},{},{},{},{},{},-1,-1,-1,-1\n'.format(frame_idx, identity, bbox_top,
                                                                                    bbox_left, bbox_w, bbox_h))
                
                if(opt.mode == "gt"):
                    persons_inf = group_frame[frame_idx+1]
                    bbox_gt_xyxy = list(map(lambda x: x.xywh_to_xyxy(), persons_inf))
                    identities_gt = list(map(lambda x: x.track_id, persons_inf))
                    draw_boxes(im0, bbox_gt_xyxy, identities_gt)

            else:
                deepsort.increment_ages()

        

    
    # for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)

    #     # Inference
    #     t1 = time_synchronized()
    #     pred = model(img, augment=opt.augment)[0]

    #     embed(header='debug pred')

    #     # Apply NMS

    #     # Returns: list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    #     pred = non_max_suppression(
    #         pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #     t2 = time_synchronized()

    #     embed(header='debug pred after nms')


        
    #     # Example pred after nms: [tensor([[444.50000,  33.65625, 492.00000, 113.87500,   0.76025,  58.00000]], device='cuda:0')]

    #     # img0 shape: 1080 x 1920 x 3
    #     # s: 384x640

    #     # Process detections
    #     for i, det in enumerate(pred):  # detections per image

    #         # det example: tensor([[444.50000,  33.65625, 492.00000, 113.87500,   0.76025,  58.00000]], device='cuda:0')
    #         if webcam:  # batch_size >= 1
    #             p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    #         else:
    #             '''
    #             p: path to vid
    #             s: size (384 x 640)
    #             im0: ndarray (1080 x 1920 x 3)
    #             '''
    #             p, s, im0 = path, '', im0s

    #         s += '%gx%g ' % img.shape[2:]  # print string  '384x640'

    #         # 'inference/output/NVR-CH01_S20210608-084648_E20210608-084709.mp4'
    #         save_path = str(Path(out) / Path(p).name)
            
    #         embed(header='debug det')

    #         if det is not None and len(det):
    #             # Rescale boxes from img_size to im0 size (from 384x640 to 1080x1920x3)
    #             det[:, :4] = scale_coords(
    #                 img.shape[2:], det[:, :4], im0.shape).round()

    #             # Print results
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

    #             xywh_bboxs = []
    #             confs = []

    #             embed(header = 'debug scale box')
    #             # Adapt detections to deep sort input format
    #             for *xyxy, conf, cls in det:
    #                 '''
    #                 *xyxy: tensor(1334., device='cuda:0') tensor(65., device='cuda:0') tensor(1476., device='cuda:0') tensor(306., device='cuda:0')
    #                 conf: tensor(0.76025, device='cuda:0')
    #                 cls: tensor(58., device='cuda:0')
    #                 '''
    #                 # to deep sort format
    #                 x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy) # convert to x center, y center, w, h
    #                 xywh_obj = [x_c, y_c, bbox_w, bbox_h]
    #                 xywh_bboxs.append(xywh_obj)
    #                 confs.append([conf.item()])

    #             xywhs = torch.Tensor(xywh_bboxs)    # tensor([[1405.00000,  185.50000,  142.00000,  241.00000]]) (1 x 4)
    #             confss = torch.Tensor(confs)        # tensor([[0.76025]]) (1 x 1)
                
    #             # pass detections to deepsort

    #             # update tracker, return tensor of xy_xy + track id 
    #             outputs = deepsort.update(xywhs, confss, im0)

    #             embed(header='debug input for deepsort')

    #             # draw boxes for visualization
    #             if len(outputs) > 0 and opt.mode == "predict":
    #                 bbox_xyxy = outputs[:, :4]
    #                 identities = outputs[:, -1]
    #                 draw_boxes(im0, bbox_xyxy, identities)
    #                 # to MOT format
    #                 tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

    #                 # Write MOT compliant results to file
    #                 if save_txt:
    #                     for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
    #                         bbox_top = tlwh_bbox[0]
    #                         bbox_left = tlwh_bbox[1]
    #                         bbox_w = tlwh_bbox[2]
    #                         bbox_h = tlwh_bbox[3]
    #                         identity = output[-1]
    #                         with open(txt_path, 'a') as f:
    #                             f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
    #                                                         bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
    #             if(opt.mode == "gt"):
    #                 persons_inf = group_frame[frame_idx+1]
    #                 bbox_gt_xyxy = list(map(lambda x: x.xywh_to_xyxy(), persons_inf))
    #                 identities_gt = list(map(lambda x: x.track_id, persons_inf))
    #                 draw_boxes(im0, bbox_gt_xyxy, identities_gt)

    #         else:
    #             deepsort.increment_ages()

    #         # Print time (inference + NMS)
    #         print('%sDone. (%.3fs)' % (s, t2 - t1))

    #         # Stream results
    #         if show_vid:
    #             cv2.imshow(p, im0)
    #             if cv2.waitKey(1) == ord('q'):  # q to quit
    #                 raise StopIteration

    #         # Save results (image with detections)
    #         if save_vid:
    #             if vid_path != save_path:  # new video
    #                 vid_path = save_path
    #                 if isinstance(vid_writer, cv2.VideoWriter):
    #                     vid_writer.release()  # release previous video writer
    #                 if vid_cap:  # video
    #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                 else:  # stream
    #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                     save_path += '.mp4'

    #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #             vid_writer.write(im0)



    '''
    ===========================================================================================
    '''

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/checkpoints/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--mode", type=str, default="predict", help='options: predict|gt')

    parser.add_argument("--frame_dir", type=str, default = '/data.local/hangd/data_vtx/frames_data/test/NVR-CH01_S20210607-102303_E20210607-102433')
    parser.add_argument("--det_pred_dir", type=str, default = '/data.local/all/hainp/yolov5_deep_sort/deep_sort_copy/track_dataset/NVR-CH01_S20210607-102303_E20210607-102433/ensemble')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
                        
    with torch.no_grad():
        detect(args)
