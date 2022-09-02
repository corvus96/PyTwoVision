Search.setIndex({docnames:["index","module_compute","module_datasets_loader","module_image_process","module_input_output","module_models","module_recognition","module_stereo","module_utils"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:55},filenames:["index.rst","module_compute.rst","module_datasets_loader.rst","module_image_process.rst","module_input_output.rst","module_models.rst","module_recognition.rst","module_stereo.rst","module_utils.rst"],objects:{"py2vision.compute":{error_compute:[1,0,0,"-"],yolov3_calculus:[1,0,0,"-"]},"py2vision.compute.error_compute":{re_projection_error:[1,1,1,""]},"py2vision.compute.yolov3_calculus":{YoloV3Calculus:[1,2,1,""]},"py2vision.compute.yolov3_calculus.YoloV3Calculus":{bbox_ciou:[1,3,1,""],bbox_giou:[1,3,1,""],bbox_iou:[1,3,1,""],best_bboxes_iou:[1,3,1,""],centroid2minmax:[1,3,1,""],decode:[1,3,1,""],loss:[1,3,1,""],minmax2centroid:[1,3,1,""],nms:[1,3,1,""],postprocess_boxes:[1,3,1,""]},"py2vision.datasets_loader":{yolov3_dataset_generator:[2,0,0,"-"]},"py2vision.datasets_loader.yolov3_dataset_generator":{YoloV3DatasetGenerator:[2,2,1,""]},"py2vision.datasets_loader.yolov3_dataset_generator.YoloV3DatasetGenerator":{delete_bad_annotation:[2,3,1,""],load_annotations:[2,3,1,""],parse_annotation:[2,3,1,""],preprocess_true_boxes:[2,3,1,""],random_crop:[2,3,1,""],random_horizontal_flip:[2,3,1,""],random_translate:[2,3,1,""]},"py2vision.image_process":{frame_decorator:[3,0,0,"-"],resize:[3,0,0,"-"],resize_with_bbox:[3,0,0,"-"],rotate:[3,0,0,"-"],split_pair:[3,0,0,"-"]},"py2vision.image_process.frame_decorator":{Frame:[3,2,1,""],FrameDecorator:[3,2,1,""]},"py2vision.image_process.frame_decorator.Frame":{apply:[3,3,1,""],img:[3,4,1,""]},"py2vision.image_process.frame_decorator.FrameDecorator":{apply:[3,3,1,""],frame:[3,4,1,""]},"py2vision.image_process.resize":{Resize:[3,2,1,""]},"py2vision.image_process.resize.Resize":{apply:[3,3,1,""]},"py2vision.image_process.resize_with_bbox":{ResizeWithBBox:[3,2,1,""]},"py2vision.image_process.resize_with_bbox.ResizeWithBBox":{apply:[3,3,1,""]},"py2vision.image_process.rotate":{Rotate:[3,2,1,""]},"py2vision.image_process.rotate.Rotate":{apply:[3,3,1,""]},"py2vision.image_process.split_pair":{SplitPair:[3,2,1,""]},"py2vision.image_process.split_pair.SplitPair":{apply:[3,3,1,""]},"py2vision.input_output":{camera:[4,0,0,"-"],vision_system:[4,0,0,"-"]},"py2vision.input_output.camera":{Camera:[4,2,1,""]},"py2vision.input_output.camera.Camera":{calibrate:[4,3,1,""],get_parameters:[4,3,1,""],take_photos:[4,3,1,""]},"py2vision.input_output.vision_system":{VisionSystem:[4,2,1,""]},"py2vision.input_output.vision_system.VisionSystem":{image_pipeline:[4,3,1,""],realtime_or_video_pipeline:[4,3,1,""]},"py2vision.models":{models_manager:[5,0,0,"-"],yolov3_model:[5,0,0,"-"],yolov3_tiny_model:[5,0,0,"-"]},"py2vision.models.blocks":{backbone_block:[5,0,0,"-"]},"py2vision.models.blocks.backbone_block":{BackboneStrategy:[5,2,1,""],darknet19_tiny:[5,2,1,""],darknet53:[5,2,1,""]},"py2vision.models.blocks.backbone_block.darknet19_tiny":{build:[5,3,1,""]},"py2vision.models.blocks.backbone_block.darknet53":{build:[5,3,1,""]},"py2vision.models.layers":{batch_normalization_layer:[5,0,0,"-"],conv2d_bn_leaky_relu_layer:[5,0,0,"-"],residual_layer:[5,0,0,"-"],upsample_layer:[5,0,0,"-"]},"py2vision.models.layers.batch_normalization_layer":{BatchNormalization:[5,2,1,""]},"py2vision.models.layers.batch_normalization_layer.BatchNormalization":{call:[5,3,1,""]},"py2vision.models.layers.conv2d_bn_leaky_relu_layer":{conv2d_bn_leaky_relu_layer:[5,1,1,""]},"py2vision.models.layers.residual_layer":{residual_layer:[5,1,1,""]},"py2vision.models.layers.upsample_layer":{UpsampleLayer:[5,2,1,""]},"py2vision.models.layers.upsample_layer.UpsampleLayer":{call:[5,3,1,""]},"py2vision.models.models_manager":{ModelManager:[5,2,1,""],ModelManagerInterface:[5,2,1,""]},"py2vision.models.models_manager.ModelManager":{build_yolov3:[5,3,1,""],build_yolov3_tiny:[5,3,1,""],model1:[5,4,1,""],model2:[5,4,1,""]},"py2vision.models.models_manager.ModelManagerInterface":{build_yolov3:[5,5,1,""],build_yolov3_tiny:[5,5,1,""]},"py2vision.models.yolov3_model":{BuildYoloV3:[5,2,1,""]},"py2vision.models.yolov3_model.BuildYoloV3":{call:[5,3,1,""]},"py2vision.models.yolov3_tiny_model":{BuildYoloV3Tiny:[5,2,1,""]},"py2vision.models.yolov3_tiny_model.BuildYoloV3Tiny":{call:[5,3,1,""]},"py2vision.recognition":{detection_mode:[6,0,0,"-"],selector:[6,0,0,"-"],yolov3_detector:[6,0,0,"-"]},"py2vision.recognition.detection_mode":{DetectImage:[6,2,1,""],DetectRealTime:[6,2,1,""],DetectRealTimeMP:[6,2,1,""],DetectVideo:[6,2,1,""],DetectionMode:[6,2,1,""]},"py2vision.recognition.detection_mode.DetectImage":{detect:[6,3,1,""],prepare_input:[6,3,1,""],show:[6,3,1,""]},"py2vision.recognition.detection_mode.DetectRealTime":{detect:[6,3,1,""],prepare_input:[6,3,1,""],show:[6,3,1,""]},"py2vision.recognition.detection_mode.DetectRealTimeMP":{detect:[6,3,1,""],multi_process_initialization:[6,3,1,""],postprocess_mp:[6,3,1,""],predict_bbox_mp:[6,3,1,""],prepare_input:[6,3,1,""],show:[6,3,1,""]},"py2vision.recognition.detection_mode.DetectVideo":{detect:[6,3,1,""],prepare_input:[6,3,1,""],show:[6,3,1,""]},"py2vision.recognition.detection_mode.DetectionMode":{camera_input:[6,3,1,""],detect:[6,3,1,""],draw:[6,3,1,""],postprocess_boxes:[6,3,1,""],pre_process:[6,3,1,""],predict:[6,3,1,""],prepare_input:[6,3,1,""],show:[6,3,1,""]},"py2vision.recognition.selector":{NeuralNetwork:[6,2,1,""],Recognizer:[6,2,1,""]},"py2vision.recognition.selector.Recognizer":{evaluate:[6,3,1,""],get_model:[6,3,1,""],inference:[6,3,1,""],print_model:[6,3,1,""],restore_weights:[6,3,1,""],train:[6,3,1,""],train_using_weights:[6,3,1,""]},"py2vision.recognition.yolov3_detector":{ObjectDetectorYoloV3:[6,2,1,""]},"py2vision.recognition.yolov3_detector.ObjectDetectorYoloV3":{build_model:[6,3,1,""],conv_tensors:[6,4,1,""],evaluate:[6,3,1,""],gpus:[6,4,1,""],inference:[6,3,1,""],input_shape:[6,4,1,""],model:[6,4,1,""],model_name:[6,4,1,""],num_class:[6,4,1,""],print_summary:[6,3,1,""],restore_weights:[6,3,1,""],train:[6,3,1,""],train_step:[6,3,1,""],validate_step:[6,3,1,""],version:[6,4,1,""]},"py2vision.stereo":{match_method:[7,0,0,"-"],standard_stereo:[7,0,0,"-"],stereo_builder:[7,0,0,"-"]},"py2vision.stereo.match_method":{Matcher:[7,2,1,""],MatcherStrategy:[7,2,1,""],StereoSGBM:[7,2,1,""]},"py2vision.stereo.match_method.Matcher":{match:[7,3,1,""],strategy:[7,4,1,""]},"py2vision.stereo.match_method.StereoSGBM":{match:[7,3,1,""]},"py2vision.stereo.standard_stereo":{StandardStereo:[7,2,1,""],StandardStereoBuilder:[7,2,1,""]},"py2vision.stereo.standard_stereo.StandardStereo":{calibrate:[7,3,1,""],camL:[7,4,1,""],camR:[7,4,1,""],fish_eye:[7,4,1,""],get_stereo_maps:[7,3,1,""],print_parameters:[7,3,1,""],rectify:[7,3,1,""],take_dual_photos:[7,3,1,""]},"py2vision.stereo.standard_stereo.StandardStereoBuilder":{camL:[7,4,1,""],camR:[7,4,1,""],estimate_3D_points:[7,3,1,""],estimate_depth_map:[7,3,1,""],estimate_disparity_colormap:[7,3,1,""],find_epilines:[7,3,1,""],get_product:[7,3,1,""],match:[7,3,1,""],post_process:[7,3,1,""],pre_process:[7,3,1,""],reset:[7,3,1,""],stereo_maps_path:[7,4,1,""]},"py2vision.stereo.stereo_builder":{StereoController:[7,2,1,""],StereoSystemBuilder:[7,2,1,""]},"py2vision.stereo.stereo_builder.StereoController":{compute_3D_map:[7,3,1,""],compute_3D_points:[7,3,1,""],compute_disparity:[7,3,1,""],compute_disparity_color_map:[7,3,1,""],get_epilines:[7,3,1,""],pre_process_step:[7,3,1,""]},"py2vision.utils":{annotations_helper:[8,0,0,"-"],annotations_parser:[8,0,0,"-"],draw:[8,0,0,"-"],label_utils:[8,0,0,"-"]},"py2vision.utils.annotations_helper":{AnnotationsHelper:[8,2,1,""]},"py2vision.utils.annotations_helper.AnnotationsHelper":{"export":[8,3,1,""],shuffle:[8,3,1,""],split:[8,3,1,""]},"py2vision.utils.annotations_parser":{AnnotationsFormat:[8,2,1,""],Parser:[8,2,1,""],XmlParser:[8,2,1,""],YoloV3AnnotationsFormat:[8,2,1,""]},"py2vision.utils.annotations_parser.XmlParser":{parse:[8,3,1,""]},"py2vision.utils.draw":{draw_bbox:[8,1,1,""],draw_lines:[8,1,1,""]},"py2vision.utils.label_utils":{class2index:[8,1,1,""],index2class:[8,1,1,""],label_map:[8,1,1,""],read_class_names:[8,1,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:staticmethod"},terms:{"1xn":1,"2007_000027":[],"2nd":0,"3xn":1,"4x4":[4,7],"640x480":7,"abstract":6,"boolean":[2,4,5,6,7,8],"byte":7,"case":[5,6,7],"class":[0,1,2,3,4,5,6,7,8],"default":[4,5,6,7,8],"direcci\u00f3n":0,"doll\u00e1r":0,"estereosc\u00f3pica":0,"export":[4,7,8],"final":[6,7],"float":[1,6,7,8],"function":[0,4,7],"import":[3,4,5,7,8],"importaci\u00f3n":[],"int":[4,8],"k\u00f6nigshof":0,"long":0,"m\u00f3dulo":[],"mart\u00edn":0,"new":[0,2,3,5,7],"p\u00e1g":0,"recci\u00f3n":[],"return":[1,2,3,4,5,6,7,8],"static":5,"t\u00e9cnica":0,"true":[2,4,5,6,7,8],"try":[4,8],"visi\u00f3n":0,"while":6,But:[4,7],For:[2,5],RMS:4,The:[0,1,4,5,6,7,8],Then:1,There:2,Useful:1,Using:0,__call__:5,__init__:5,a_calibration_image_folder_path:4,a_path_or_a_name_for_calibration_it_doesn:4,abl:3,about:0,abov:[1,5],abs:0,absent:7,absolut:1,accept:7,accesado:0,accord:7,accordingli:7,accur:1,acm:0,activ:5,actual:6,adam:6,adapt:2,add:4,adding:1,addit:[4,5],adher:[4,7],adjust:7,adl:[],adopt:[],adrian:0,advanc:0,after:[1,7],against:5,aggreg:0,algorithm:[0,1,4,5,6],align:[],alkjrhglkopdjnt6shgbphthymct0cumjg:6,all:[1,3,4,5,6,7,8],allow:[0,2,4,6,7,8],alpha:7,alreadi:[0,7],also:[1,7],alter:3,although:5,alwai:[5,6,7],amount:[1,4,7],an_image_path:3,analysi:0,anchor:[1,2,6],anchor_per_scal:[2,6],angl:3,ani:[4,7],anno:8,anno_format:8,anno_help:8,anno_out_fil:[4,8],anno_out_full_path:8,annot:[0,2,4,6],annotations_form:[4,8],annotations_help:8,annotations_output_nam:8,annotations_pars:8,annotations_path:[2,8],annotationsformat:8,annotationshelp:8,anoth:[3,7,8],api:8,appli:[0,1,3,4,5,6,7],applic:[0,4],approach:1,aracil:0,architectur:[2,5],area:7,aren:8,arg:5,argument:[1,4,5,6,7,8],arithmet:1,around:4,arr:7,arrai:[1,2,3,4,5,6,7,8],arxiv:[0,1,6],ass:[4,6,8],asset:4,assist:0,atienza:0,attribut:6,attributeerror:7,augment:2,autoencod:0,autom:0,automat:[5,6],automodul:[],avail:[],averag:1,axi:5,backbon:5,backbone_block:5,backbone_net:5,backboneblock:5,backbonestrategi:5,background:[],bad_annot:2,barnard:0,base:[1,3,5,6,8],basenam:4,basi:[],batch:[1,2,5,6],batch_normalization_lay:5,batch_siz:[2,6],batchnorm:5,bbox:[1,2,6,8],bbox_ciou:1,bbox_giou:1,bbox_iou:1,becaus:[2,7],befor:[1,5,6,7],begin:6,behavior:5,being:3,belong:2,belongi:0,best:[1,6,7],best_bboxes_i:1,beta_constraint:5,beta_initi:5,beta_regular:5,better:1,between:[1,4,6,7,8],bgr:7,bharatsingh430:[],big:7,binari:4,birchfield:7,bitgener:8,black:7,blob:[],block:[0,7],blur:4,blurri:7,bmatrix:[],bmp:4,bond:1,both:[1,4,7],bottom:3,bound:[1,2,3,4,6,8],box:[1,2,3,4,6,8],boxes1:1,boxes2:1,branch:5,build:[5,6,7],build_model:6,build_yolov3:5,build_yolov3_tini:5,builder:0,buildyolov3:5,buildyolov3tini:5,built:5,calcul:0,calibr:[0,1,7],calibratecamera:1,california:0,call:[3,5,7,8],cam_left:[4,7],cam_right:[4,7],came:5,camera:[0,1,3,6,7],camera_input:6,caml:7,camr:7,can:[0,1,2,3,4,5,6,7],cannot:5,capabl:4,captur:4,cast:5,categori:1,center:[1,4,5,7],centroid2minmax:1,centroid:1,certain:1,challeng:0,chang:[1,2,6,7],channel:[1,6,7],chapter:[],charg:[1,5],check:[5,7],checkpoint:6,checkpoint_path:6,chepoint:6,chessboard:[4,7],choic:1,ciou:1,circl:[4,7],ckpt:6,class2index:8,class_file_nam:[2,4,6,8],class_nam:8,classes_fil:[4,6,8],classes_out_fil:8,classes_output_nam:8,classif:2,client:7,clip:7,close:6,cnn:0,coco:[4,8],code:[1,6,7],coeffici:1,collect:5,color:[4,6,7,8],column:[4,7],com:[0,4,6],combin:0,come:5,common:[3,5,7],compar:1,compat:[2,5,6,8],complet:[1,6],compon:[3,7],comput:[0,4,5,7],compute_3d_map:7,compute_3d_point:7,compute_dispar:7,compute_disparity_color_map:7,concept:3,concret:[3,5,7,8],conf_loss:6,confer:0,confid:[1,8],configur:7,connect:7,consid:[1,7],consol:[0,4,7,8],constructor:7,consum:7,contain:[1,2,4,5,6,7],content:[],context:[5,7],contol:7,contour:7,contrast:[4,7],contrib:0,contribut:0,control:[0,2,6],conv2d:5,conv2d_bn_leaky_relu_lay:5,conv:1,conv_lbbox:5,conv_mbbox:5,conv_output:1,conv_sbbox:5,conv_tensor:6,convers:1,convert:[1,2,6,7,8],convolut:[0,1,5,6],coordin:[0,1,4,6,7,8],corner:[1,4,7],correct:7,correspond:[0,1,2,5,6,7,8],corvus96:[],cost:7,cpp:7,crawler:[],creat:[1,5,6,7,8],creation:5,crop:2,cs231n:0,csur:0,current:[2,7,8],custom:5,cv2:[4,7],dai:0,darker:4,darknet19:5,darknet19_tini:5,darknet53:5,darrel:0,data:[2,4,6,8],data_augment:2,datafram:8,dataset:[0,6,8],datasets_load:2,deal:8,debug:6,decim:7,declar:[5,7,8],decod:1,decor:3,deep:0,defin:[3,4,5,6,7],deleg:[3,7],delet:2,delete_bad_annot:2,demand:5,dembi:0,demultipli:4,dens:0,depend:[2,5,6],depth:[0,5,7],depthwis:5,deriv:7,descript:1,desouza:0,detail:[0,1,5,7],detect:[0,1,2],detectimag:6,detection_mod:6,detectionmod:6,detector:6,detectrealtim:6,detectrealtimemp:6,detectvideo:6,determinar:0,dev:[],develop:2,dib:4,dict:[5,8],dictionari:8,did:5,didn:[4,7],differ:[3,4,5,6,7,8],digraph:[],dim:6,dimens:[0,1,2,3,4,5,6,8],dimension:1,direct:1,directli:5,directori:[4,7,8],disabl:7,discard:[1,4,6],disp_12_max_diff:[4,7],dispar:[0,4,7],disparity_map_exampl:[],dist:1,dist_coeff:7,distanc:0,distort:[1,7],distribuir:[],distribut:[],divis:7,doc:[],doc_stat:[],doe:7,doesn:6,doi:0,don:[4,7],done:1,download:[4,6],downsampl:[4,5,7],downsample_for_match:4,draw:[0,4,6,7],draw_bbox:8,draw_lin:8,drive:0,dst:[],dst_path:8,dtype:5,dure:[4,7],dynam:[5,7],e_matrix:7,each:[1,2,4,5,6,7,8],easi:8,easiest:[],ecosystem:5,edg:[4,7],edit:0,educ:0,effect:3,egin:[],either:5,element:[6,7,8],embed:0,empti:[4,6,8],emul:[4,7],encompass:6,end:[],enorm:5,enough:7,entir:6,epicent:0,epilin:[7,8],epoch:6,epsilon:5,equat:[],error:[0,4,7],error_comput:1,escena:0,esenti:7,especi:7,especif:8,estim:0,estimate_3d_point:7,estimate_depth_map:7,estimate_disparity_colormap:7,estruc:0,etc:5,evalu:[0,2,6],even:[2,4,7],everi:0,everingham:0,exact:8,exampl:0,except:[4,5,8],execut:[5,7],exist:[4,5,7],expect:[1,4,6],explicitli:5,export_fil:[4,7],export_file_nam:[4,7],express:1,extern:4,extrem:7,extrins:[1,4,7],exttt:[],eye:[4,7],f_matrix:7,fact:6,factor:[4,7],fals:[2,4,5,6,7,8],farhadi:0,faster:2,favor:4,featur:0,febrero:0,field:[4,7],figur:[],file:[2,4,6,7,8],file_path:8,filter:[1,4,5,7],filter_num1:5,filter_num2:5,filters_shap:5,final_fram:6,find:[1,4,7,8],find_epilin:7,first:[1,4,5,6,7,8],fischler:0,fish:[4,7],fish_ey:[4,7],fishey:[4,7],fisheye_camera:4,fit:8,flexibl:0,flip:2,focal:[1,4,7],folder:[4,6,7],follow:[4,5,7],forc:[4,7],format:[1,4,7,8],formula:[],forum:[],found:[1,4,5,7],frame:[0,1,3,4,6,7],frame_decor:3,frame_transform:3,framedecor:3,framel:7,framer:7,frames_data:6,free:7,freenod:[],freez:5,from:[1,2,3,4,5,6,7,8],full:[7,8],fulli:0,fundament:7,gamma_constraint:5,gamma_initi:5,gamma_regular:5,gan:0,gao:0,gener:[1,2,5,6,8],get:[0,3,5,6,7,8],get_epilin:7,get_model:[4,6],get_paramet:4,get_product:7,get_stereo_map:7,gil:0,giou:1,giou_loss:6,girshick:0,github:0,given:[1,5],global_step:6,going:6,gonna:6,good:7,googl:[],gool:0,got:[],gpu:6,gpu_nam:6,grade:4,gradient:0,grai:7,graph:5,grayscal:4,greater:7,greather:1,grid:1,ground:[1,2],group:[5,8],gt_box:3,guid:[0,5],h5py:0,hand:4,handl:5,hang:[],happen:4,hardwar:[0,4],hariharan:0,hartlei:0,has:[1,4,5,7],have:[0,4,5,6,7,8],haven:7,height:[1,3,4,6,7],help:[7,8],helper:0,here:[0,4,5,8],hide:8,higher:6,highlight:[],hiperparamet:6,his:[4,7],homogen:[0,6,7,8],homogeneous_point:[6,8],horizont:[2,3],host:2,hous:1,how:[0,1,7],howev:[4,6,8],html:0,http:[0,1,4,5,6],huang:0,huge:7,ident:0,identifi:[4,8],ids:8,ieee:0,imag:[0,1,2,4,5,6,7,8],image_data:6,image_left__dim:7,image_path:[6,8],image_pipelin:4,image_process:3,image_right_dim:7,images_left:7,images_left_path:7,images_path:[4,8],images_right:7,images_right_path:7,images_to_ram:2,img1:8,img2:8,img:3,imgpoint:1,implement:[0,5,8],implicitli:7,improv:[0,1,4,6,7],imread:3,inc:0,includ:7,increment:0,index2class:8,index:[0,8],indic:5,individu:7,infer:[0,5],info:5,inform:0,infti:[],init_scop:5,initi:[4,6],initialit:[6,7],inner:[4,7],input:[0,1,2,3,5,6,7],input_channel:5,input_data:5,input_imag:3,input_lay:5,input_output:[4,6,7],input_path:6,input_s:[1,4,6],input_shap:[2,5,6],input_spec:5,insid:[1,6],instanc:[4,5,6,7],instead:7,int_:[],integ:[1,2,3,4,5,6,7],integr:5,intellig:0,inteng:[],interact:[6,7],interest:7,interfac:[3,4,5,6,7,8],intermedi:7,intern:[0,8],intersect:1,interv:7,intrins:[1,4,7],introduc:2,invalid:7,invoc:5,iou:1,iou_threshold:[1,4,6],isn:[4,6],issn:0,issu:[],iter:[2,4,7],its:[2,3,4,5,6,7],itsc:0,join:[4,8],journal:0,jpe:4,jpeg:4,jpg:[4,7],just:[5,6,7],kaehler:0,kar:0,kei:8,kera:[0,5],kernel:5,keyword:5,know:[5,7],kwarg:5,label:[0,1,2,6],label_map:8,label_util:8,larg:[4,5,7],larger:[4,7],last:[1,5,6],layer:[0,1,6],lead:[4,7],leakag:[4,7],leaki:5,leakyrelu:5,learn:[0,6],left:[3,4,7],left_camera:4,left_camera_calibr:4,left_disp:7,left_imag:7,left_indoor_photo_5:4,left_plant_1:4,len:1,length:1,lenth:[4,7],leran:6,less:[1,4,6,7],level:[6,8],librari:0,lighter:4,like:[0,1,5,6,7,8],lin:0,line:[2,8],link_yolov3_weight:4,list:[1,2,3,5,6,7,8],live:[4,5],lmbda:[4,7],load:[4,6,7,8],load_annot:2,loader:0,local:6,locat:[2,6],log:6,log_dir:6,logic:5,london:0,loss:[1,6],loss_thresh:1,lost:7,low:[4,7],lower:1,lr_end:6,lr_init:6,machin:0,made:6,madrid:0,magic:4,mai:[1,5],mail:[],maintain:7,make:[2,4,5,7],manag:0,mani:1,mantic:0,manual:5,map:[0,2,4,6,7],margin:7,mask:5,masking_and_pad:5,master:0,match:[6,7],match_method:[4,7],matcher:[4,7],matcherstrategi:7,math:[],mathrm:[],matplotlib:[],matric:1,matrix:[1,4,7],max_bbox_per_scal:[2,6],max_disp:[4,7],maxdepth:[],maximum:[1,6,7],maxvalu:[],mean:[1,5,7],meant:5,measur:0,mechan:[1,2,7],media:[0,4,6],mediat:5,medium:[4,5],meet:1,member:[],merg:4,metadata:5,method:[1,2,4,5,6,7,8],metric:7,min_disp:[4,7],minimum:7,minmax2centroid:1,minmax:1,minu:7,mix:5,mkdir:[4,8],mode:[0,3,5,7],mode_nam:6,model1:5,model2:5,model:[0,1,2,4,6,8],model_manag:5,model_nam:6,modelmanag:5,modelmanagerinterfac:5,models_manag:5,modifi:5,modul:[0,1,2,4,8],modulo:[],momentum:5,more:[0,1,2,4,5,7],moving_mean_initi:5,moving_variance_initi:5,mp4:4,mtx:1,multi:[0,6],multi_process_initi:6,multipl:7,multipli:7,multiprocess:6,must:[1,2,5,7,8],name:[4,5,6,7,8],nan:[2,6],necessari:[3,4,5],need:[1,2,4,6,7],neighbor:7,net:6,network:[0,1,2,5,8],neural:[0,1,2,5],neural_network:6,neuralnetwork:6,next:[1,6,7],niqu:0,nms:[1,4,6],nms_method:[4,6],nois:[1,4,7],non:[1,6,7],none:[3,4,5,6,7,8],norm:1,normal:[4,5,7],note:[1,4,5,7],noviembr:[],num_class:[1,5,6],num_photo:[4,7],number:[1,2,4,5,6,7],number_of_image_channel:7,numdispar:7,numpi:[0,5],nx1:1,nx3:1,object:[0,1,2,3,5,7,8],objectdetectoryolov3:[4,6],objpoint:1,obtain:[0,1,4],oct:0,odd:7,offer:5,offset:1,onc:7,one:[1,3,5,7],one_camera_paramet:7,ones:5,onli:[4,5,6,7],open:0,opencv:[0,6,7],oper:[3,5,6,7],ops:5,optim:6,option:5,order:[1,7],org:[0,1,5,6],origin:[1,2,4,6,7],original_fram:6,original_imag:[1,6],ork:[],oserror:[4,7],other:[4,5,6],otherwis:[2,4,6,7],otsu:4,otsu_thresh_invers:4,ouput:6,our:[1,4],out:[],output:[0,1,5,6,7,8],output_path:[4,6],output_s:7,over:1,overridden:5,own:[0,5,7],packag:[0,5,6],pad:5,page:[],pair:[3,7],panda:[0,8],paper:[1,6],paquet:[],para:0,param:[4,6],paramet:[1,2,3,4,5,6,7,8],parameter:4,pars:8,parse_annot:2,parser:0,particular:7,pascal:[0,8],pass:[4,5,7,8],path:[2,4,6,7,8],pattern:[0,4,7],pattern_s:[4,7],pattern_typ:[4,7],pbtxt:8,pcess:6,pdf:[1,6],pearson:0,penalti:7,pep:[],per:[2,4,6,7],percentag:[7,8],perform:[1,7],person:7,photo:[4,7],photo_1:7,php:0,physic:4,physiolog:0,pictur:7,piec:1,pip:0,pipelin:6,pixel:[4,7],pjreddi:[4,6],plane:7,plu:7,png:4,point:[1,4,7,8],polici:0,pomar:0,pose:0,posit:[0,5,7],possibl:7,post:[4,7],post_process:[4,7],post_process_match:4,postprocess:6,postprocess_box:[1,6],postprocess_mp:6,practic:0,pre:[6,7],pre_filter_cap:[4,7],pre_process:[6,7],pre_process_step:7,precis:5,pred:1,pred_bbox:[1,6],predict:[1,5,6],predict_bbox_mp:6,predicted_data:6,prefilt:7,prefiltercap:7,prefix:[4,7],prefix_nam:[4,7],preocess:6,prepar:2,prepare_input:6,preprocess:2,preprocess_true_box:2,present:2,preserv:7,press:6,previou:5,previous:6,primit:6,princip:[4,7],print:[6,7,8],print_model:6,print_output:8,print_paramet:7,print_summari:6,prob_loss:6,probabl:[1,2],proceed:0,process:[0,4,6,7],processed_fram:6,processing_tim:6,produc:7,product:7,program:7,project:[0,1,6],projectpoint:1,proport:8,provid:[4,5,6,7],proyecto:0,pts1:8,pts2:8,puent:0,purpos:6,put:4,py2vis:[1,2,3,4,5,6,7,8],pyramid:[0,7],python:[0,5],pytwovis:[],pyyaml:0,q_matrix:4,quantiti:5,queue:6,quick:0,radial:7,rais:[4,7,8],ram:2,randint:5,random:[5,8],random_crop:2,random_horizontal_flip:2,random_st:8,random_transl:2,randomst:8,rang:[1,4,7],rate:6,raw:[],re_projection_error:1,read:6,read_class_nam:8,real:[4,6,7],realtim:[0,4,6],realtime_or_video_pipelin:4,reappli:5,reason:[2,5,7],receiv:1,recogn:[4,6],recognit:[0,2,4,5],recomend:[4,7],recommend:[5,6,7],recov:6,rectangle_color:[4,6,8],rectif:[0,7],rectifi:[4,7],redmon:0,reduc:5,reescal:3,refer:[1,7,8],region:7,regular:[4,7],reilli:0,reinforc:0,relev:[4,7],reli:5,relu:5,ren:0,repres:[1,3,4,8],reproject:[0,7],requir:[1,2,7],reserv:5,reset:7,residu:[0,5],residual_lay:5,resiz:[1,2,3,4,6],resize_dim:4,resize_with_bbox:3,resizewithbbox:3,resnet:5,resolut:7,resourc:5,respect:[1,7],respons:[1,7],rest:8,restore_weight:[4,6],result:[0,3,7],retain:7,rgb:7,right:[3,4,7],right_camera:4,right_camera_calibr:4,right_disp:7,right_imag:7,right_indoor_photo_5:4,right_plant_1:4,rms:7,robot:0,rodrigu:1,role:5,root:6,rot:7,rotat:[1,3,4,7],row:[4,6,7,8],royal:0,rule:5,run:[5,7],runtim:7,rvec:1,sadwindows:7,salscheid:0,sampl:7,save:[2,4,6,7,8],save_all_checkpoint:6,save_dir:4,save_dir_left:7,save_dir_right:7,save_only_best_model:6,savedmodel:5,sbs:3,scalar:5,scale:[1,2,5,6,7],scene:0,scharstein:0,score:[1,4,6],score_threshold:[1,4,6],search:[1,6],second:[5,6,7,8],see:[0,1,7],seed:5,segment:[0,2],selector:[0,4,5],self:5,semant:0,sensit:[4,7],sensorial:0,separ:5,sequenc:7,set:[2,5,7,8],setter:7,sgbm:[0,4],shafiekhani:0,shape:[1,2,5,6,7],shelham:0,shift:[2,7],should:[5,6,7,8],show:[4,6,7,8],show_confid:8,show_label:8,show_window:4,shown:7,shuffl:8,side:[3,7],sigma:[1,4,7],signatur:8,simpl:[4,7],sinc:5,singl:5,sion:0,sistema:0,size:[1,2,5,6,7],skeleton:6,slice:3,small:[4,5,7],smooth:7,smoother:7,societi:0,soft:[1,4,6],some:[3,4,5,7],someon:8,sometim:[2,7],somewher:7,sort:7,sourc:[1,2,3,4,5,6,7,8],space:0,special:[2,5],specif:[5,7],specifi:7,speckl:7,speckle_rang:[4,7],speckle_window_s:[4,7],split:[3,8],split_pair:3,splitpair:3,sqrt:[],squar:[1,6],src:[],ssd:6,stack:[3,5],standard:0,standard_stereo:[4,7],standardstereo:[4,7],standardstereobuild:7,start:0,state:5,step:[4,6,7],stereo:[0,1,3,4],stereo_build:7,stereo_maps_path:[4,7],stereo_match:7,stereo_pair_fishey:4,stereo_paramet:7,stereocontrol:7,stereomap:[4,7],stereopi:0,stereopi_v2_quick_start_guid:0,stereosgbm:[4,7],stereosgbm_mode_hh:7,stereosystembuild:7,stiller:0,store:5,strategi:[5,7],stream:[4,6],stride:[1,2,6],string:[1,2,4,5,6,8],structur:[4,6,7],subclass:5,subject:5,submodul:5,subsampl:7,summari:6,sun:0,support:[5,7],suppress:1,supress:[1,6],survei:0,system:[0,1,7],szeliski:0,t_matter:4,tabl:[],take:[2,4,5,7,8],take_dual_photo:7,take_photo:4,target:6,target_s:3,taxonomi:0,tech:0,techniqu:0,technolog:0,tensor:[1,5,6],tensorflow:[0,4,6,8],tesorflow:6,test:[4,6,8],test_anno_fil:8,test_annotations_path:6,test_class:8,test_dataset:[4,8],test_dataset_gener:8,test_input_s:6,test_posit:4,text:[4,8],text_color:[4,6,8],textur:[4,7],tflite:5,tfmot:5,than:[1,4,5,6,7],thei:7,them:[1,2],theori:0,thi:[0,1,2,4,5,6,7,8],third:[5,7],those:[1,6,7,8],three:[1,5,7],threshold:[1,4],through:[4,7],time:[5,6,7],tini:[5,6],tion:0,titl:0,to_generator_test:[4,8],toctre:[],togeth:1,tomasi:7,too:[4,7],tool:[0,5],top:3,torr:0,total:6,total_loss:6,track:8,train:[0,1,2,4,5,6,8],train_annotations_path:6,train_class:[],train_percentag:8,train_step:6,train_using_weight:6,trainabl:5,training_percen:8,tran:7,transac:0,transfer:6,transform:[0,1,4,7,8],translat:[1,4,7],transmiss:4,transport:0,tridimension:0,true_box:2,truncat:7,truth:1,tupl:[1,2,3,4,5,6,7,8],tura:0,tutori:[6,7],tvec:1,tvsnet:0,two:[0,5,6,7,8],txt:[2,4,6,8],type:[1,2,8],typeerror:8,typic:[4,6,7],underli:5,understand:1,uninstal:0,union:1,uniqueness_ratio:[4,7],unit:7,updat:6,upsampl:5,upsample_lay:5,upsamplelay:5,url:[4,6],usa:0,use:[0,4,6,7],use_checkpoint:6,used:[0,4,5,6,7],useful:1,uses:[5,7],usg:6,using:[0,1,3,5,6,7],util:0,valid:[1,4,6,7],validate_step:6,valu:[1,4,5,7],valueerror:[4,8],van:0,vari:2,variabl:[1,4,5,6],variat:[0,7],variou:5,vector:1,version:[0,1,5,6,7],vertic:3,via:[4,5,7],vid:6,video:[4,6],videocaptur:6,view:[0,1,4,7],vis_si:4,visibl:7,vision:[0,4],vision_system:4,visionsystem:4,visit:8,visitor:8,visual:0,voc2012:0,voc:[0,8],vol:0,wai:[2,3,4,7,8],wanna:6,want:[4,6,7],warmup_epoch:6,webcam:[4,6],webp:4,weight:[4,5,6],weights_01:6,weights_fil:[4,6],weights_path:6,wget:[0,4],what:1,wheatston:0,when:[1,2,4,5,6,7,8],where:[1,2,4,5,6,7,8],whether:5,which:[0,1,2,4,5,6,7,8],width:[1,3,4,7],wifi:4,wiki:0,william:0,win:7,window:[4,6],window_s:[4,7],winn:0,wit:[],within:[5,7],without:[4,6,7],wls:7,won:[2,4,6,7,8],work:[3,5,7],work_dir:[4,8],workshop:0,world:[1,4,7],wrap:[3,5],write:[0,6],wrong:4,wrt:[],www:[0,5],xmax:1,xmin:1,xml:[4,7,8],xml_path:[4,8],xmlparser:8,yield:7,ymax:1,ymin:1,yolo:[0,2,6],yolo_coco_class:[],yolov3:[0,1,4,5,8],yolov3_calculu:1,yolov3_dataset_gener:2,yolov3_detector:[4,6],yolov3_model:5,yolov3_tini:6,yolov3_tiny_model:5,yolov3annotationsformat:8,yolov3calculu:1,yolov3datasetgener:[2,6],yolov3model:5,yolov3tinymodel:5,you:[1,4,6,7,8],your:[0,6,7],zero:[1,5,7],zhang:0,zisserman:0,zoom:7},titles:["Welcome to Py2vision\u2019s documentation!","Compute","Datasets loader","Image process","Inputs and Outputs","Tensorflow models","Object detection (Recognition)","Stereo Vision","Utilities"],titleterms:{"function":8,The:3,algorithm:7,annot:8,archiev:0,block:5,builder:7,calcul:1,calibr:4,camera:4,code:8,comput:1,control:7,could:0,dataset:2,depend:0,depth:[],detect:6,document:0,draw:8,epicent:3,error:1,exampl:[6,7,8],get:[],helper:8,how:[3,4,5],imag:3,implement:[4,6,7],indic:[],infer:6,input:4,instal:0,label:8,layer:5,like:4,loader:2,manag:5,mode:6,model:5,modul:3,network:6,note:[],object:6,output:4,parser:8,posit:4,process:3,py2vis:0,pytwovis:[],recognit:6,refer:0,reproject:1,selector:6,sgbm:7,singl:4,standard:7,stereo:7,support:[],system:4,tabl:[],tensorflow:5,thi:3,transform:3,two:[],use:[3,5],util:8,vision:7,welcom:0,what:0,yolo:1,yolov3:6,you:0}})