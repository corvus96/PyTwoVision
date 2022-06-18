import os
import numpy as np
import tensorflow as tf
import shutil
import json
import time

from tensorflow.keras.utils import plot_model

from recognition.selector import NeuralNetwork
from pytwovision.models.models_manager import ModelManager
from pytwovision.models.blocks.backbone_block import BackboneBlock
from pytwovision.models.blocks.backbone_block import darknet53, darknet19_tiny
from pytwovision.compute.yolov3_calculus import YoloV3Calculus
from pytwovision.utils.label_utils import read_class_names
from pytwovision.image_process.frame_decorator import Frame
from pytwovision.image_process.resize_with_bbox import ResizeWithBBox
from pytwovision.datasets_loader.yolov3_dataset_generator import YoloV3DatasetGenerator

class ObjectDetectorYoloV3(NeuralNetwork):
    """Made of an Yolo network model and a dataset generator.
    Arguments:
        backbone (class): Contains resnet network which can be version 1 or 2
        data_path (str): beginning by project root this is the path 
                         where is save your dataset; example: "dataset/drinks".
        train_labels_csv (str): contains the path of train labels csv.
        input_shape: A tuple with dims shape (height, weight, channels). 
        layers (int): Number of feature extraction layers of SSD head after backbone.
        threshold (float): Labels IoU threshold.
        normalize (bool): Use normalized predictions.   
        batch_size (int): Batch size during training.
        loss (str): Use focal and smooth L1 loss functions "focal-smooth-l1" 
        or smoth L1 "smooth-l1" even L1 "l1".
        class_threshold (float): Class probability threshold (>= is an object).
        iou_threshold (float): NMS IoU threshold.
        soft_nms (bool): Use soft NMS or not.
        save_dir_path: Directory for saving model and weights
    Attributes:
        ssd (model): SSD network model
        train_generator: Multi-threaded data generator for training
    """
    def __init__(self, model_name, num_class, data_path, train_labels_csv, input_shape=(416, 416, 3), version="yolov3", training=False, save_dir_path="weights", batch_size=4, threshold=0.6, normalize=False, class_threshold=0.5, iou_threshold=0.2, soft_nms=False, gpu_name=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_labels_csv = train_labels_csv
        self.batch_size = batch_size
        self.normalize = normalize
        self.threshold = threshold
        self.save_dir = save_dir_path
        self.class_threshold = class_threshold
        self.soft_nms = soft_nms
        self.iou_threshold = iou_threshold
        self.model = None
        self.train_generator = None
        self.num_class = num_class
        self.version = version
        
        if gpu_name == None:
            self.gpus = tf.config.experimental.list_physical_devices('GPU')
            if len(self.gpus) > 0:
                print(f'GPUs {self.gpus}')
                try: tf.config.experimental.set_memory_growth(self.gpus[0], True)
                except RuntimeError: pass

        if self.version == "yolov3":
            backbone_net = BackboneBlock(darknet53())
            model_manager = ModelManager()
            self.conv_tensors = model_manager.build_yolov3(model_name, backbone_net, self.num_class)(input_shape)
        
        if self.version == "yolov3_tiny":
            backbone_net = BackboneBlock(darknet19_tiny())
            model_manager = ModelManager()
            self.conv_tensors = model_manager.build_yolov3_tiny(model_name, backbone_net, self.num_class)(input_shape)
        
        else: 
            versions = ["yolov3", "yolov3_tiny"]
            raise ValueError("yolo_version only can be: {}".format(", ".join(versions)))
        
        self.model = self.build_model(self.conv_tensors, training)
    
    def build_model(self, conv_tensors, training):
        """Build the complete yolo model and return model instance"""
        output_tensors = []
        input_layer = conv_tensors[-1]
        self.training = training
        compute = YoloV3Calculus()
        for i, conv_tensor in enumerate(conv_tensors[:-1]):
            pred_tensor = compute.decode(conv_tensor, self.num_class, i)
            if self.training: output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        return tf.keras.Model(input_layer, output_tensors)

    def train(self, train_annotations_path, test_annotations_path, class_file_name, 
                checkpoint_path="checkpoints", use_checkpoint=False, warmup_epochs=2, 
                epochs=100, log_dir="logs", save_only_best_model=True, save_all_checkpoints=False):
        """Train an ssd network.
        Arguments:
            epochs (int): Number of epochs to train.
            loss (str): Use focal and smooth L1 loss functions "focal-smooth-l1" 
            or smoth L1 "smooth-l1" even L1 "l1".
        """
        training = True
        if self.training == False:
            self.model = self.build_model(self.conv_tensors, training)

        if os.path.exists(log_dir): shutil.rmtree(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)
        train_set = YoloV3DatasetGenerator(train_annotations_path, class_file_name)
        test_set = YoloV3DatasetGenerator(test_annotations_path, class_file_name)

        steps_per_epoch = len(train_set)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = epochs * steps_per_epoch

        if use_checkpoint:
            try:
                self.model.load_weights(checkpoint_path)
            except ValueError:
                assert Exception("Shapes are incompatible between model and weights")
                #use_checkpoint = False
        
        checkpoint_path_splited = os.path.split(checkpoint_path)
        if len(checkpoint_path_splited[0]) == 0: 
            try:
                os.mkdir("checkpoints")
            except OSError as error:
                print(error)
            checkpoint_folder = "checkpoints"
        else: 
            checkpoint_folder = checkpoint_path_splited[0]
        optimizer = tf.keras.optimizers.Adam()

        validate_writer = tf.summary.create_file_writer(log_dir)

        mAP_model = self.build_model(self.conv_tensors, training) # create second model to measure mAP

        best_val_loss = 1000 # should be large at start
        for epoch in range(epochs):
            for image_data, target in train_set:
                results = self.train_step(image_data, target, optimizer)
                cur_step = results[0]%steps_per_epoch
                print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                    .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))

            if len(test_set) == 0:
                print("configure TEST options to validate model")
                self.model.save_weights(os.path.join(checkpoint_folder, self.model._name))
                continue
            
            count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
            for image_data, target in test_set:
                results = self.validate_step(image_data, target)
                count += 1
                giou_val += results[0]
                conf_val += results[1]
                prob_val += results[2]
                total_val += results[3]
            # writing validate summary data
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
                tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
                tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
                tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
            validate_writer.flush()
                
            print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
                format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

            if save_all_checkpoints and not save_only_best_model:
                save_directory = os.path.join(checkpoint_folder, self.model._name+"_val_loss_{:7.2f}".format(total_val/count))
                self.model.save_weights(save_directory)
            if save_only_best_model and best_val_loss>total_val/count:
                save_directory = os.path.join(checkpoint_folder, self.model._name)
                self.model.save_weights(save_directory)
                best_val_loss = total_val/count
            if not save_only_best_model and not save_all_checkpoints:
                save_directory = os.path.join(checkpoint_folder, self.model._name)
                self.model.save_weights(save_directory)
        
        # measure mAP of trained custom model
        mAP_model.load_weights(save_directory) # use keras weights
        self.evaluate(mAP_model, test_set, class_file_name, 0.3, 0.45)

    
    def train_step(self, image_data, target, optimizer, lr_init=1e-4, lr_end=1e-6):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0
            compute = YoloV3Calculus()
            # optimizing process
            grid = 3 if not (self.version == "yolov3_tiny") else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute.loss(pred, conv, *target[i], i, self.num_class)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:# and not TRAIN_TRANSFER:
                lr = self.global_steps / self.warmup_steps * lr_init
            else:
                lr = lr_end + 0.5 * (lr_init - lr_end)*(
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self.writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)
            self.writer.flush()
            
        return self.global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def validate_step(self, image_data, target):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0
            compute = YoloV3Calculus()
            # optimizing process
            grid = 3 if not (self.version == "yolov3_tiny") else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute.loss(pred, conv, *target[i], i, self.num_class)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def restore_weights(self, weights_file):
        """Load previously trained model weights
        Arguments: 
            weights_file: beginning by project root this is the path 
                                   where is save your weights; example: "weights/weights_01.h5"
        """
        tf.keras.backend.clear_session() # used to reset layer names
        # load Darknet original weights to TensorFlow model
        range1 = 75 if self.version == "yolov3" else 13
        range2 = [58, 66, 74] if self.version == "yolov3" else [9, 12]

        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' %i
                else:
                    conv_layer_name = 'conv2d'
                    
                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' %j
                else:
                    bn_layer_name = 'batch_normalization'
                
                conv_layer = self.model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = self.model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'


    def inference(self, image):
        """ Apply inference with trained model
        Arguments:
            image (tensor): contains an image converted in tensor.
        """
        image = np.expand_dims(image, axis=0)
        classes, offsets = self.ssd.predict(image)
        image = np.squeeze(image, axis=0)
        classes = np.squeeze(classes)
        offsets = np.squeeze(offsets)
        return image, classes, offsets


    def evaluate(self, model, dataset, classes_file, score_threshold=0.05, iou_threshold=0.50, test_input_size=416):
        min_overlap = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
        num_class = read_class_names(classes_file)

        ground_truth_dir_path = 'mAP/ground-truth'
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

        if not os.path.exists('mAP'): os.mkdir('mAP')
        os.mkdir(ground_truth_dir_path)

        print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')

        gt_counter_per_class = {}
        for index in range(dataset.num_samples):
            ann_dataset = dataset.annotations[index]

            original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
            num_bbox_gt = len(bboxes_gt)

            bounding_boxes = []
            for i in range(num_bbox_gt):
                class_name = num_class[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox = xmin + " " + ymin + " " + xmax + " " +ymax
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
            with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
                json.dump(bounding_boxes, outfile)

        gt_classes = list(gt_counter_per_class.keys())
        # sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)

        times = []
        json_pred = [[] for i in range(n_classes)]
        for index in range(dataset.num_samples):
            ann_dataset = dataset.annotations[index]

            image_name = ann_dataset[0].split('/')[-1]
            original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
            frame = Frame(image)
            image = ResizeWithBBox(frame).apply([test_input_size, test_input_size])
            image_data = image[np.newaxis, ...].astype(np.float32)

            t1 = time.time()
            model.predict()
            
            t2 = time.time()
            
            times.append(t2-t1)
            
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = YoloV3Calculus().postprocess_boxes(pred_bbox, original_image, test_input_size, score_threshold)
            bboxes = YoloV3Calculus().nms(bboxes, iou_threshold, method='nms')

            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = num_class[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox = xmin + " " + ymin + " " + xmax + " " +ymax
                json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        for class_name in gt_classes:
            json_pred[gt_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
            with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
                json.dump(json_pred[gt_classes.index(class_name)], outfile)

        # Calculate the AP for each class
        sum_AP = 0.0
        ap_dictionary = {}
        # open file to store the results
        with open("mAP/results.txt", 'w') as results_file:
            results_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                # Load predictions of that class
                predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
                predictions_data = json.load(open(predictions_file))

                # Assign predictions to ground truth objects
                nd = len(predictions_data)
                tp = [0] * nd # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, prediction in enumerate(predictions_data):
                    file_id = prediction["file_id"]
                    # assign prediction to ground truth object if any
                    #   open ground-truth with that file_id
                    gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load prediction bounding-box
                    bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    # assign prediction as true positive/don't care/false positive
                    if ovmax >= min_overlap:# if ovmax > minimum overlap
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                    else:
                        # false positive
                        fp[idx] = 1

                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                #print(tp)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                #print(rec)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                #print(prec)

                ap, mrec, mprec = self.__voc_ap(rec, prec)
                sum_AP += ap
                text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

                rounded_prec = [ '%.3f' % elem for elem in prec ]
                rounded_rec = [ '%.3f' % elem for elem in rec ]
                # Write to results.txt
                results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

                print(text)
                ap_dictionary[class_name] = ap

            results_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / n_classes

            text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
            results_file.write(text + "\n")
            print(text)
            
            return mAP*100

    def __voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab:  for i=numel(mpre)-1:-1:1
                                    mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

    def print_summary(self):
        """Print network summary for debugging purposes."""
        self.backbone.summary()
        self.model.summary()
        plot_model(self.backbone,
                    to_file="backbone.png",
                    show_shapes=True)
