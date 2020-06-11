# 基于MobileNet的垃圾分类助手 

#### 介绍
我是厦门大学16级人工智能系本科生，这是我的毕业设计。主要内容是一个使用Tensorflow Lite框架和MobileNet模型的垃圾分类助手。它最大的创新点是不需要服务器联网，可以直接在移动端本地进行识别。


#### 第一步 数据准备
数据下载地址为：https://pan.baidu.com/s/1ZO2Nn1cveyMxjcPmLNNzow  提取码：47d1

这个是一个40个类的垃圾图片集合。下载后解压。对应类别如下：
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"

#### 第二步 重新训练
git一下训练模型代码和Android相关代码
本来想上传的 但是好像有点多而且挺麻烦 我就不上传了
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

cd tensorflow-for-poets-2
然后我们只需要关注 scripts文件夹和Android文件夹和tflite文件夹
scripts是关于重新训练的代码
tflite存放模型文件和数据集
Android用来存放Android相关的代码

1. 考虑到测试手机性能还不赖，我们选择mobilenet_v1_1.0_224这个版本作为我们的预编译模型。

2. scripts目录下的retrain.py是我们需要关注的，这个代码目前仅支持Inception_v3和Mobilenet两种预编译模型，默认的训练模型为Inception_v3。我们使用的是Mobilenet模型

3.  重新训练模型命令
python -m scripts.retrain \　--learning_rate=0.01 \ 　
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=tf_files/models/ \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=tf_files/data \
  --architecture=mobilenet_1.0_224

命令解释：
--image_dir 为数据地址
--output_labels 最后训练生成模型的标签
-output_graph 最后训练生成的模型
--model_dir 命令启动后，预编译模型的下载地址
--how_many_training_steps 训练步数，不指定的话默认为4000
--bottleneck_dir用来把top层的训练数据缓存成文件
--learning_rate 学习率
 此外，还有些参数可以根据需要进行调整：
   --testing_percentage 把图片按多少比例划分出来当做test数据，默认为10
   --validation_percentage 把图片按多少比例划分出来当做validation数据，默认为10，这两个值设置完后，training数据占比80%
   --eval_step_interval 多少步训练后进行一次评估，默认为10
   --train_batch_size 一次训练的图片数，默认为100
   --validation_batch_size 一次验证的图片数，默认为100
   --random_scale 给定一个比例值，然后随机扩大训练图片的大小，默认为0
   --random_brightness 给定一个比例值，然后随机增强或减弱训练图片的明亮程度，默认为0
   --random_crop 给定一个比例值，然后随机裁剪训练图片的边缘值，默认为0 
   
4 检验效果
    我们用Mobilenet_1.0_224进行训练，完成后找一张图片看看是否能正确识别：
python -m scripts.label_image \
  --graph=tf_files/retrained_graph.pb  \
  --image=tf_files/data/0/1.jpg

结果：
Evaluation time (1-image): 1.010s

快餐盒 (score=0.62305)
塑料制品 (score=0.22490)
饮料瓶 (score=0.14169)


#### 第五步 转换模型格式

1.  pb格式不能运行在TFlite，需要将pb转换为lite文件
2.  转换命令 因为我们的模型是MobileNet 命令如下
toco \
  --input_file=tf_files/retrained_graph.pb \
  --output_file=tf_files/optimized_graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT
3.  最后optimized_graph.lite即是我们要移植到android上的模型文件啦。



#### Android TFLite

1.  使用Android studio 引入tflite目录下的代码 我们只需要关注ImageClassifier.java类
2.  导入模型
把模型文件和标签文件放入目录下
tflite/app/src/main/assets/mobilenet.lite 
tflite/app/src/main/assets/mobilenet.txt
3.   修改ImageClassifier.java类
/** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "mobilenet.lite";

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "mobilenet.txt";

  static final int DIM_IMG_SIZE_X = 224; //若是inception，改成299
  static final int DIM_IMG_SIZE_Y = 224; //若是inception，改成299

4.  运行观看效果
连上手机后，点击“Run”->"Run app"即会部署app到手机上，此时任何被摄像头捕获的图片都会按照标签里的40个分类进行识别排名


