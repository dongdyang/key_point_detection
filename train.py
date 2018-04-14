import matplotlib as mpl

mpl.use('Agg')  # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorpack.dataflow.remote import RemoteDataZMQ
from scipy.ndimage import maximum_filter, gaussian_filter
from collections import namedtuple
import math
import itertools
from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from networks import get_network
import common
from common import CocoPairsNetwork, CocoPairs, CocoPart


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_imgfile(path, width, height):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def _quantize_img(npimg):
    npimg_q = npimg + 1.0
    npimg_q /= (2.0 / 2 ** 8)
    # npimg_q += 0.5
    npimg_q = npimg_q.astype(np.uint8)
    return npimg_q


def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

    return npimg


def _get_scaled_img(npimg, scale, target_size):
    get_base_scale = lambda s, w, h: max(target_size[0] / float(w), target_size[1] / float(h)) * s
    img_h, img_w = npimg.shape[:2]

    if scale is None:
        if npimg.shape[:2] != (target_size[1], target_size[0]):
            # resize
            npimg = cv2.resize(npimg, target_size)
        return [npimg], [(0.0, 0.0, 1.0, 1.0)]
    elif isinstance(scale, float):
        # scaling with center crop
        base_scale = get_base_scale(scale, img_w, img_h)
        npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
        ratio_x = (1. - target_size[0] / float(npimg.shape[1])) / 2.0
        ratio_y = (1. - target_size[1] / float(npimg.shape[0])) / 2.0
        roi = _crop_roi(npimg, ratio_x, ratio_y, target_size)
        return [roi], [(ratio_x, ratio_y, 1. - ratio_x * 2, 1. - ratio_y * 2)]
    elif isinstance(scale, tuple) and len(scale) == 2:
        # scaling with sliding window : (scale, step)
        base_scale = get_base_scale(scale[0], img_w, img_h)
        base_scale_w = target_size[0] / (img_w * base_scale)
        base_scale_h = target_size[1] / (img_h * base_scale)
        npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
        window_step = scale[1]
        rois = []
        infos = []
        for ratio_x, ratio_y in itertools.product(np.arange(0., 1.01 - base_scale_w, window_step),
                                                  np.arange(0., 1.01 - base_scale_h, window_step)):
            roi = _crop_roi(npimg, ratio_x, ratio_y, target_size)
            rois.append(roi)
            infos.append((ratio_x, ratio_y, base_scale_w, base_scale_h))
        return rois, infos
    elif isinstance(scale, tuple) and len(scale) == 3:
        # scaling with ROI : (want_x, want_y, scale_ratio)
        base_scale = get_base_scale(scale[2], img_w, img_h)
        npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
        ratio_w = target_size[0] / float(npimg.shape[1])
        ratio_h = target_size[1] / float(npimg.shape[0])

        want_x, want_y = scale[:2]
        ratio_x = want_x - ratio_w / 2.
        ratio_y = want_y - ratio_h / 2.
        ratio_x = max(ratio_x, 0.0)
        ratio_y = max(ratio_y, 0.0)
        if ratio_x + ratio_w > 1.0:
            ratio_x = 1. - ratio_w
        if ratio_y + ratio_h > 1.0:
            ratio_y = 1. - ratio_h

        roi = _crop_roi(npimg, ratio_x, ratio_y, target_size)
        return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]


def _crop_roi(npimg, ratio_x, ratio_y, target_size):
    target_w, target_h = target_size
    h, w = npimg.shape[:2]
    x = max(int(w * ratio_x - .5), 0)
    y = max(int(h * ratio_y - .5), 0)
    cropped = npimg[y:y + target_h, x:x + target_w]

    cropped_h, cropped_w = cropped.shape[:2]
    if cropped_w < target_w or cropped_h < target_h:
        npblank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
        npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
    else:
        return cropped


def inference(sess, input_node, vectmap_node, heatmap_node, npimg, target_size):
    ENSEMBLE = 'addup'
    scales = None
    if npimg is None:
        raise Exception('The image is not valid. Please check your image exists.')

    if not isinstance(scales, list):
        scales = [None]

    rois = []
    infos = []

    scale = None
    roi, info = _get_scaled_img(npimg, scale, target_size)
    rois.extend(roi)
    infos.extend(info)

    logger.debug('inference+')
    pafMats, heatMats = sess.run([vectmap_node, heatmap_node], feed_dict={input_node: rois})

    logger.debug('inference-')

    output_h, output_w = heatMats.shape[1:3]
    max_ratio_w = max_ratio_h = 10000.0
    for info in infos:
        max_ratio_w = min(max_ratio_w, info[2])
        max_ratio_h = min(max_ratio_h, info[3])
    mat_w, mat_h = int(output_w / max_ratio_w), int(output_h / max_ratio_h)
    resized_heatMat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
    resized_pafMat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
    resized_cntMat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
    resized_cntMat += 1e-12

    for heatMat, pafMat, info in zip(heatMats, pafMats, infos):
        w, h = int(info[2] * mat_w), int(info[3] * mat_h)
        heatMat = cv2.resize(heatMat, (w, h))
        pafMat = cv2.resize(pafMat, (w, h))
        x, y = int(info[0] * mat_w), int(info[1] * mat_h)

        if ENSEMBLE == 'average':
            # average
            resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0, -y):, max(0, -x):, :]
            resized_pafMat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
            resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1
        else:
            # add up
            resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
            resized_pafMat[max(0, y):y + h, max(0, x):x + w, :] += pafMat[max(0, -y):, max(0, -x):, :]
            resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1

    if ENSEMBLE == 'average':
        heatMat = resized_heatMat / resized_cntMat
        pafMat = resized_pafMat / resized_cntMat
    else:
        heatMat = resized_heatMat
        pafMat = resized_pafMat / (np.log(resized_cntMat) + 1)

    humans = PoseEstimator.estimate(heatMat, pafMat)
    return humans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet', help='model name')
    # parser.add_argument('--datapath', type=str, default='/media/ydd/dong/coco_dataset/annotations')
    # parser.add_argument('--imgpath', type=str, default='/media/ydd/dong/coco_dataset/')

    parser.add_argument('--datapath', type=str, default='/home/ydd/Desktop/ali_dataset/annotations/')
    parser.add_argument('--imgpath', type=str, default='/home/ydd/Desktop/ali_dataset/')

    parser.add_argument('--imageTest', type=str,
                        default='/home/ydd/Desktop/ali_dataset/val2018/0a4e49305744d9120d25bed3d421f417.jpg')

    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=str, default='0.001')
    parser.add_argument('--modelpath', type=str,
                        default='/home/ydd/Desktop/ali_cloth_tf4/models/tf-openpose-models-2018-4/')
    parser.add_argument('--logpath', type=str, default='/home/ydd/Desktop/ali_cloth_tf4/logs/tf-openpose-log-2018-4/')

    parser.add_argument('--checkpoint', type=str, default='/home/ydd/Desktop/ali_cloth_tf4/models/tf-openpose-models-2018-4/mobilenet_batch:5_lr:0.01_gpus:1_368x368_-200')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)

    args = parser.parse_args()

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2', 'mobilenet_try3',
                      'hybridnet_try']:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
    input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3),
                                name='image')
    vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 38), name='vectmap')
    heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 19), name='heatmap')
    # prepare data
    if not args.remote_data:
        df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath)
    else:
        # transfer inputs from ZMQ
        df = RemoteDataZMQ(args.remote_data, hwm=3)
    # enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
    q_inp, q_heat, q_vect = input_node, heatmap_node, vectmap_node  # enqueuer.dequeue()

    #df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath)
    # df_valid.reset_state()
    validation_cache = []

    # val_image = get_sample_images(args.input_width, args.input_height)
    # logger.info('tensorboard val image: %d' % len(val_image))
    logger.info(q_inp)
    logger.info(q_heat)
    logger.info(q_vect)

    # define model for multi-gpu
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect,
                                                                                                                args.gpus)

    output_vectmap = []
    output_heatmap = []
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    outputs = []
    # for gpu_id in range(args.gpus):
    #    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        net, pretrain_path, last_layer = get_network(args.model, q_inp_split[0])
        vect, heat = net.loss_last()
        output_vectmap.append(vect)
        output_heatmap.append(heat)
        outputs.append(net.get_output())

        l1s, l2s = net.loss_l1_l2()
        for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
            loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[0],
                                    name='loss_l1_stage%d_tower%d' % (idx, 0))
            loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[0],
                                    name='loss_l2_stage%d_tower%d' % (idx, 0))
            losses.append(tf.reduce_mean([loss_l1, loss_l2]))

        last_losses_l1.append(loss_l1)
        last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    # with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
    # define loss
    total_loss = tf.reduce_sum(losses) / args.batchsize
    total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
    total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
    total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])

    # define optimizer
    step_per_epoch = 3000 // args.batchsize
    global_step = tf.Variable(0, trainable=False)
    if ',' not in args.lr:
        starter_learning_rate = float(args.lr)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=10000, decay_rate=0.33, staircase=True)
    else:
        lrs = [float(x) for x in args.lr.split(',')]
        boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    # tf.summary.scalar("queue_size", enqueuer.size())
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    # sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    # sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    # train_img = tf.summary.image('training sample', sample_train, 4)
    # valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        training_name = '{}_batch:{}_lr:{}_gpus:{}_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.gpus,
            args.input_width, args.input_height,
            args.tag
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        
        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, args.checkpoint)
            logger.info('Restore from checkpoint...Done')
        '''
        elif pretrain_path:
            logger.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(pretrain_path, sess, False)
            logger.info('Restore pretrained weights...Done')
        '''
        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(args.logpath + training_name, sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        # enqueuer.set_coordinator(coord)
        # enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        df.reset_state()

        while True:
            for dp in df.get_data():
                feed = dict(zip([input_node, heatmap_node, vectmap_node], dp))

                _, gs_num = sess.run([train_op, global_step], feed_dict=feed)

                #print("DONEONDONEODNOEN")

                if gs_num > step_per_epoch * args.max_epoch:
                    break

                if gs_num - last_gs_num >= 10:
                    train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, lr_val, summary = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, learning_rate,
                         merged_summary_op
                         ], feed_dict=feed)  # queue_size        enqueuer.size()

                    # log of training loss / accuracy
                    batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                    logger.info(
                        'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, loss_ll_paf=%g, loss_ll_heat=%g' % (
                            gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss,
                            train_loss_ll,
                            train_loss_ll_paf, train_loss_ll_heat))  # , queue_size
                    last_gs_num = gs_num

                    file_writer.add_summary(summary, gs_num)

                    #print("Summany Once")

                if (gs_num - last_gs_num2) % 100 == 0:
                    # save weights
                    saver.save(sess, os.path.join(args.modelpath, training_name), global_step=global_step)

                    '''
                    average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                    total_cnt = 0

                    if len(validation_cache) == 0:
                        for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                            validation_cache.append((images_test, heatmaps, vectmaps))
                        df_valid.reset_state()
                        del df_valid
                        df_valid = None
                    print("Done0_0")
                    # log of test accuracy

                    for images_test, heatmaps, vectmaps in validation_cache:
                        lss, lss_ll, lss_ll_paf, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                            [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat, output_vectmap,
                             output_heatmap],
                            feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                        )
                        average_loss += lss * len(images_test)
                        average_loss_ll += lss_ll * len(images_test)
                        average_loss_ll_paf += lss_ll_paf * len(images_test)
                        average_loss_ll_heat += lss_ll_heat * len(images_test)
                        total_cnt += len(images_test)

                    logger.info('validation(%d) %s loss=%f, loss_ll=%f, loss_ll_paf=%f, loss_ll_heat=%f' % (
                    total_cnt, training_name, average_loss / total_cnt, average_loss_ll / total_cnt,
                    average_loss_ll_paf / total_cnt, average_loss_ll_heat / total_cnt))
                    last_gs_num2 = gs_num
                    '''
                    # sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                    # outputMat = sess.run(
                    #    outputs,
                    # feed_dict={q_inp: np.array((sample_image + val_image)*(args.batchsize // 16))}
                    #    feed_dict={q_inp: np.array((sample_image)*(args.batchsize // 16))}
                    # )
                    # pafMat, heatMat = outputMat[:, :, :, 19:], outputMat[:, :, :, :19]
                    # print("Done0_1")

                    # sample_results = []
                    # for i in range(len(sample_image)):
                    #    test_result = CocoPose.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                    #    test_result = cv2.resize(test_result, (640, 640))
                    #    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    #    sample_results.append(test_result)

                    # test_results = []
                    # for i in range(len(val_image)):
                    #    test_result = CocoPose.display_image(val_image[i], heatMat[len(sample_image) + i], pafMat[len(sample_image) + i], as_numpy=True)
                    #    test_result = cv2.resize(test_result, (640, 640))
                    #    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    #   test_results.append(test_result)

                    # save summary
                    '''
                    summary = sess.run(merged_validate_op, feed_dict={
                        valid_loss: average_loss / total_cnt,
                        valid_loss_ll: average_loss_ll / total_cnt,
                        valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                        valid_loss_ll_heat: average_loss_ll_heat / total_cnt  # ,
                        # sample_valid: test_results,
                        # sample_train: sample_results
                    })
                    file_writer.add_summary(summary, gs_num)
                    # break
                    '''
                '''
                if (gs_num - last_gs_num2) % 30 == 0:
                    image = read_imgfile(args.imageTest, None, None)
                    #
                    target_size = (args.input_width, args.input_height)
                    humans = inference(sess, input_node, vectmap_node, heatmap_node, image, target_size)

                    image = draw_humans(image, humans, imgcopy=False)
                    # cv2.imshow('tf-pose-estimation result', image)
                    # cv2.waitKey()

                    import matplotlib.pyplot as plt

                    fig = plt.figure()
                    a = fig.add_subplot(2, 2, 1)
                    a.set_title('Result')
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    break
                '''
        print("Done1")
        # saver.save(sess, os.path.join(args.modelpath, training_name), global_step=global_step)

    # sess.run(enqueuer.close())

    print("Done")
    logger.info('optimization finished. %f' % (time.time() - time_started))













































class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])



class PoseEstimator:
    heatmap_supress = False
    heatmap_gaussian = False
    adaptive_threshold = False

    NMS_Threshold = 0.15
    Local_PAF_Threshold = 0.2
    PAF_Count_Threshold = 5
    Part_Count_Threshold = 4
    Part_Score_Threshold = 4.5

    PartPair = namedtuple('PartPair', [
        'score',
        'part_idx1', 'part_idx2',
        'idx1', 'idx2',
        'coord1', 'coord2',
        'score1', 'score2'
    ], verbose=False)

    def __init__(self):
        pass

    @staticmethod
    def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
        under_threshold_indices = plain < threshold
        plain[under_threshold_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def estimate(heat_mat, paf_mat):
        if heat_mat.shape[2] == 19:
            heat_mat = np.rollaxis(heat_mat, 2, 0)
        if paf_mat.shape[2] == 38:
            paf_mat = np.rollaxis(paf_mat, 2, 0)

        if PoseEstimator.heatmap_supress:
            heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(19, 1, 1)
            heat_mat = heat_mat - heat_mat.min(axis=2).reshape(19, heat_mat.shape[1], 1)

        if PoseEstimator.heatmap_gaussian:
            heat_mat = gaussian_filter(heat_mat, sigma=0.5)

        if PoseEstimator.adaptive_threshold:
            _NMS_Threshold = max(np.average(heat_mat) * 4.0, PoseEstimator.NMS_Threshold)
            _NMS_Threshold = min(_NMS_Threshold, 0.3)
        else:
            _NMS_Threshold = PoseEstimator.NMS_Threshold

        # extract interesting coordinates using NMS.
        coords = []  # [[coords in plane1], [....], ...]
        for plain in heat_mat[:-1]:
            nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
            coords.append(np.where(nms >= _NMS_Threshold))

        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)

        # merge pairs to human
        # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
        humans = [Human([pair]) for pair in pairs_by_conn]
        while True:
            merge_items = None
            for k1, k2 in itertools.combinations(humans, 2):
                if k1 == k2:
                    continue
                if k1.is_connected(k2):
                    merge_items = (k1, k2)
                    break

            if merge_items is not None:
                merge_items[0].merge(merge_items[1])
                humans.remove(merge_items[1])
            else:
                break

        # reject by subset count
        humans = [human for human in humans if human.part_count() >= PoseEstimator.PAF_Count_Threshold]

        # reject by subset max score
        humans = [human for human in humans if human.get_max_score() >= PoseEstimator.Part_Score_Threshold]

        return humans

    @staticmethod
    def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
        connection_temp = []

        cnt = 0
        for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
            for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
                score, count = PoseEstimator.get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
                cnt += 1
                if count < PoseEstimator.PAF_Count_Threshold or score <= 0.0:
                    continue
                connection_temp.append(PoseEstimator.PartPair(
                    score=score,
                    part_idx1=part_idx1, part_idx2=part_idx2,
                    idx1=idx1, idx2=idx2,
                    coord1=(x1 * rescale[0], y1 * rescale[1]),
                    coord2=(x2 * rescale[0], y2 * rescale[1]),
                    score1=heatmap[part_idx1][y1][x1],
                    score2=heatmap[part_idx2][y2][x2],
                ))

        connection = []
        used_idx1, used_idx2 = set(), set()
        for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
            # check not connected
            if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
                continue
            connection.append(candidate)
            used_idx1.add(candidate.idx1)
            used_idx2.add(candidate.idx2)

        return connection

    @staticmethod
    def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
        __num_inter = 10
        __num_inter_f = float(__num_inter)
        dx, dy = x2 - x1, y2 - y1
        normVec = math.sqrt(dx ** 2 + dy ** 2)

        if normVec < 1e-4:
            return 0.0, 0

        vx, vy = dx / normVec, dy / normVec

        xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
        ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
        xs = (xs + 0.5).astype(np.int8)
        ys = (ys + 0.5).astype(np.int8)

        # without vectorization
        pafXs = np.zeros(__num_inter)
        pafYs = np.zeros(__num_inter)
        for idx, (mx, my) in enumerate(zip(xs, ys)):
            pafXs[idx] = paf_mat_x[my][mx]
            pafYs[idx] = paf_mat_y[my][mx]

        # vectorization slow?
        # pafXs = pafMatX[ys, xs]
        # pafYs = pafMatY[ys, xs]

        local_scores = pafXs * vx + pafYs * vy
        thidxs = local_scores > PoseEstimator.Local_PAF_Threshold

        return sum(local_scores * thidxs), sum(thidxs)



