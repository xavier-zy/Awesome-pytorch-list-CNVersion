Awesome-Pytorch-list｜厉害的Pytorch项目
========================

![pytorch-logo-dark](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)

## [English Version](https://github.com/bharathgs/Awesome-pytorch-list)

## Contents｜内容
- [Awesome-Pytorch-list｜厉害的Pytorch项目](#awesome-pytorch-list%EF%BD%9C%E5%8E%89%E5%AE%B3%E7%9A%84pytorch%E9%A1%B9%E7%9B%AE)
  - [English Version](#english-version)
  - [Contents｜内容](#contents%EF%BD%9C%E5%86%85%E5%AE%B9)
  - [Pytorch & related libraries｜Pytorch & 相关库](#pytorch-related-libraries%EF%BD%9Cpytorch-%E7%9B%B8%E5%85%B3%E5%BA%93)
    - [NLP & Speech Processing｜自然语言处理 & 语音处理:](#nlp-speech-processing%EF%BD%9C%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86-%E8%AF%AD%E9%9F%B3%E5%A4%84%E7%90%86)
    - [CV｜计算机视觉:](#cv%EF%BD%9C%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)
    - [Probabilistic/Generative Libraries｜概率库和生成库:](#probabilisticgenerative-libraries%EF%BD%9C%E6%A6%82%E7%8E%87%E5%BA%93%E5%92%8C%E7%94%9F%E6%88%90%E5%BA%93)
    - [Other libraries｜其他库:](#other-libraries%EF%BD%9C%E5%85%B6%E4%BB%96%E5%BA%93)
  - [Tutorials & examples｜教程 & 示例](#tutorials-examples%EF%BD%9C%E6%95%99%E7%A8%8B-%E7%A4%BA%E4%BE%8B)
  - [Paper implementations｜论文实现](#paper-implementations%EF%BD%9C%E8%AE%BA%E6%96%87%E5%AE%9E%E7%8E%B0)
  - [Talks & conferences｜报告 & 会议](#talks-conferences%EF%BD%9C%E6%8A%A5%E5%91%8A-%E4%BC%9A%E8%AE%AE)
  - [Pytorch elsewhere ｜ Pytorch相关](#pytorch-elsewhere-%EF%BD%9C-pytorch%E7%9B%B8%E5%85%B3)
        
## Pytorch & related libraries｜Pytorch & 相关库

1. [pytorch](http://pytorch.org): Tensors and Dynamic neural networks in Python with strong GPU acceleration | 使用强GPU加速的Python张量计算和动态神经网络.

### NLP & Speech Processing｜自然语言处理 & 语音处理:

1. <kbd>1800+</kbd> [text](https://github.com/pytorch/text): 针对文本数据和NLP数据集的数据加载和抽象。
2. <kbd>1000+</kbd> [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq): Pytorch中处理seq2seq的开源框架。
3. <kbd>1000-</kbd> [anuvada](https://github.com/Sandeep42/anuvada): NLP可解释模型。
4. <kbd>1000-</kbd> [audio](https://github.com/pytorch/audio): 简单的音频I/O。
5. <kbd>1000-</kbd> [loop](https://github.com/facebookresearch/loop):  一种跨多说话者的语音生成方法。
6. <kbd>5200+</kbd> [fairseq](https://github.com/facebookresearch/fairseq-py): Facebook开发的Sequence-to-Sequence python工具包。
7. <kbd>1000-</kbd> [speech](https://github.com/awni/speech): 语音转文字的端到端模型实现。
8. <kbd>3300+</kbd> [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py): 开源神经机器翻译 http://opennmt.net.
9. <kbd>1500+</kbd> [neuralcoref](https://github.com/huggingface/neuralcoref): 在spaCy中使用神经网络实现快速共指消解。
10. <kbd>1000-</kbd> [sentiment-discovery](https://github.com/NVIDIA/sentiment-discovery): 基于规模的无监督语言模型在稳健情绪分类中的应用。
11. <kbd>2100+</kbd> [MUSE](https://github.com/facebookresearch/MUSE): 一个多语言无监督或有监督词语嵌入库。
12. <kbd>1000-</kbd> [nmtpytorch](https://github.com/lium-lst/nmtpytorch): PyTorch中的Sequence-to-Sequence框架。
13. <kbd>1000-</kbd> [pytorch-wavenet](https://github.com/vincentherrmann/pytorch-wavenet): 快速生成WaveNet的实现。
14. <kbd>1000-</kbd> [Tacotron-pytorch](https://github.com/soobinseo/Tacotron-pytorch): Tacotron: 端到端语音合成。
15. <kbd>7000+</kbd> [AllenNLP](https://github.com/allenai/allennlp): 开源NLP研究库，基于PyTorch。[http://www.allennlp.org/](https://allennlp.org)
16. <kbd>1400+</kbd> [PyTorch-NLP](https://github.com/PetrochukM/PyTorch-NLP): 为加速NLP研究设立的一个库，包含神经网络层、文本处理模块和众多数据集。 pytorchnlp.readthedocs.io
17. <kbd>1000-</kbd> [quick-nlp](https://github.com/outcastofmusic/quick-nlp): 基于FastAI的Pytorch NLP库。
18. <kbd>1200+</kbd> [TTS](https://github.com/mozilla/TTS): 文本转语音的深度学习框架。
19. <kbd>1900+</kbd> [LASER](https://github.com/facebookresearch/LASER): LASER是一个用来计算和使用多语言语句嵌入的库。
20. <kbd>1000-</kbd> [pyannote-audio](https://github.com/pyannote/pyannote-audio): 用于说话人分类的神经构建块：语音活动检测, 说话人变化检测, 说话人嵌入。
21. <kbd>1000-</kbd> [gensen](https://github.com/Maluuba/gensen): 基于大规模多任务学习的通用句子表示。
22. <kbd>1000-</kbd> [translate](https://github.com/pytorch/translate): 翻译——一个PyTorch语言库。
23. <kbd>1200+</kbd> [espnet](https://github.com/espnet/espnet): 端到端语音处理工具集。 espnet.github.io/espnet
24. <kbd>2700+</kbd> [pythia](https://github.com/facebookresearch/pythia): 源于FAIR(Facebook AI Research)的视觉与语言多模态研究的模块化框架。
25. <kbd>1200+</kbd> [UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT): 基于短语的神经无监督机器翻译。
26. <kbd>1000-</kbd> [jiant](https://github.com/jsalt18-sentence-repl/jiant): 通用文本理解模型的jiant工具包。https://jiant.info
27. <kbd>2700+</kbd> [BERT-PyTorch](https://github.com/codertimo/BERT-pytorch): Google AI 2018 BERT 的 Pytorch 实现，伴有简单注释。
28. <kbd>1600+</kbd> [InferSent](https://github.com/facebookresearch/InferSent): NLI的句子嵌入(InferSent)和训练代码。
29. <kbd>1000-</kbd> [uis-rnn](https://github.com/google/uis-rnn):无限交错状态递归神经网络(UIS-RNN)算法，能够从嘈杂的环境中分辨声音，对应论文 Fully Supervised Speaker Diarization. arxiv.org/abs/1810.04719
30. <kbd>7000+</kbd> [flair](https://github.com/zalandoresearch/flair): 一个针对最先进的NLP的简单框架。
31. <kbd>5300+</kbd> [pytext](https://github.com/facebookresearch/pytext): 基于PyTorch的自然语言建模框架。 fb.me/pytextdocs
32. <kbd>1000-</kbd> [voicefilter](https://github.com/mindslab-ai/voicefilter): 谷歌AI的VoiceFilter的非官方实现。 http://swpark.me/voicefilter
33. <kbd>1000-</kbd> [BERT-NER](https://github.com/kamalkraj/BERT-NER): 基于BERT的命名体识别(Named-Entity-Recognition)。
34. <kbd>1000-</kbd> [transfer-nlp](https://github.com/feedly/transfer-nlp): 为可复制实验管理而设计的NLP库。
35. <kbd>1000-</kbd> [texar-pytorch](https://github.com/asyml/texar-pytorch): 机器学习和文本生成工具包。 texar.io
36. <kbd>1100+</kbd> [pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi): pytorch-kaldi 是一个开发中的最先进的dnn/rnn混合语音识别系统。其DNN部分由PyTorch实现，而特征提取、标签计算和解码由kaldi工具包完成。

### CV｜计算机视觉:

1. <kbd>4200+</kbd> [pytorch vision](https://github.com/pytorch/vision): TorchVision包含流行的数据集、模型架构、计算机视觉中常用的图像变换。
2. <kbd>1000-</kbd> [pt-styletransfer](https://github.com/tymokvo/pt-styletransfer): 作为PyTorch中一个类的神经风格转移。
3. <kbd>1000-</kbd> [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch): 使用OpenFace的nn4.small2.v1.t7模型的PyTorch模块。
4. <kbd>1000-</kbd> [img_classification_pk_pytorch](https://github.com/felixgwu/img_classification_pk_pytorch): 将你的图像分类模型和最先进的模型进行快速比较 (比如DenseNet, ResNet, ...)
5. <kbd>1000-</kbd> [SparseConvNet](https://github.com/facebookresearch/SparseConvNet): 子流形稀疏卷积神经网络。
6. <kbd>1000-</kbd> [Convolution_LSTM_pytorch](https://github.com/automan000/Convolution_LSTM_pytorch): 多层卷积LSTM(长短期记忆网络)模块。
7. <kbd>3100+</kbd> [face-alignment](https://github.com/1adrianb/face-alignment): :fire: 使用pytorch构建2D和3D人脸比对库。 adrianbulat.com
8. <kbd>1000+</kbd> [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation): 语义分割。
9. <kbd>1000-</kbd> [RoIAlign.pytorch](https://github.com/longcw/RoIAlign.pytorch): PyTorch版本的RoIAlign。其实现基于crop_and_resize，支持CPU和GPU上的前向和后向。
10. <kbd>1000-</kbd> [pytorch-cnn-finetune](https://github.com/creafz/pytorch-cnn-finetune): 用PyTorch微调预训练卷积神经网络。
11. <kbd>1000-</kbd> [detectorch](https://github.com/ignacio-rocco/detectorch): Detectorch - PyTorch版detectron框架，目前仅有detectron的推断(inference)和评估(evalutaion)功能，无训练(training)功能。
12. <kbd>3400+</kbd> [Augmentor](https://github.com/mdbloice/Augmentor): 用于机器学习的图像增强库。 http://augmentor.readthedocs.io
13. <kbd>1000-</kbd> [s2cnn](https://github.com/jonas-koehler/s2cnn): Spherical CNNs：球面卷积网络的PyTorch实现。 (e.g. 全方位图像、全球信号)
14. <kbd>1000-</kbd> [PyTorchCV](https://github.com/CVBox/PyTorchCV): 基于PyTorch的计算机视觉深度学习框架。
15. <kbd>6300+</kbd> [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark): 实例分割与对象检测的快速模块化参考实现。
16. <kbd>1000-</kbd> [image-classification-mobile](https://github.com/osmr/imgclsmob): 计算机视觉卷积网络训练沙盒，包含ImageNet-1K上的与训练分类模型集合。
17. <kbd>1000-</kbd> [medicaltorch](https://github.com/perone/medicaltorch): 一个医学成像框架。http://medicaltorch.readthedocs.io
18. <kbd>3000+</kbd> [albumentations](https://github.com/albu/albumentations): 快速图像增强库和其他库的易用包装器。
19. <kbd>1000-</kbd> [kornia](https://github.com/arraiyopensource/kornia): 开源可微计算机视觉库。https://kornia.org

### Probabilistic/Generative Libraries｜概率库和生成库:

1. <kbd>1000-</kbd> [ptstat](https://github.com/stepelu/ptstat): 概率编程和统计推断。
2. <kbd>5500+</kbd> [pyro](https://github.com/uber/pyro): Python 和 PyTorch 的深度通用概率编程。 http://pyro.ai
3. <kbd>1000-</kbd> [probtorch](https://github.com/probtorch/probtorch): Probabilistic Torch是一个扩展了PyTorch的深度生成模型的库。
4. <kbd>1000-</kbd> [paysage](https://github.com/drckf/paysage): 基于Python/PyTorch的非监督学习和生成模型库。
5. <kbd>1000-</kbd> [pyvarinf](https://github.com/ctallec/pyvarinf): Python包，促进了带有变分推断的贝叶斯深度学习方法在pytorch中的应用。
6. <kbd>1000-</kbd> [pyprob](https://github.com/probprog/pyprob): 一个基于PyTorch的概率编程与推断编译的库。
7. <kbd>1000-</kbd> [mia](https://github.com/spring-epfl/mia): 一个运行针对机器学习模型的成员推理攻击的库。
8. <kbd>1000-</kbd> [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): 作为PyTorch nn.Module的扩展的ProGAN包。
9. <kbd>1000+</kbd> [botorch](https://github.com/pytorch/botorch): PyTorch中的贝叶斯优化。

### Other libraries｜其他库:

1. <kbd>1000-</kbd> [pytorch extras](https://github.com/mrdrozdov/pytorch-extras): PyTorch的额外特性。
2. <kbd>1000-</kbd> [functional zoo](https://github.com/szagoruyko/functional-zoo): PyTorch和Tensorflow的模型定义和预训练权重。
3. <kbd>1300+</kbd> [torch-sampling](https://github.com/ncullen93/torchsample): Pytorch的采样、高级训练、数据增强和实用程序。
4. <kbd>1000-</kbd> [torchcraft-py](https://github.com/deepcraft/torchcraft-py): TorchCraft的Python包装器，TorchCraft是连接Torch和StarCraft的桥梁。
5. <kbd>1000-</kbd> [aorun](https://github.com/ramon-oliveira/aorun): Aorun试图以PyTorch为后端实现类似于Keras的API。
6. <kbd>1000-</kbd> [logger](https://github.com/oval-group/logger): 机器学习记录器（logger）。
7. <kbd>1000-</kbd> [PyTorch-docset](https://github.com/iamaziz/PyTorch-docset): PyTorch离线文档，结合Dash，Zeal，Velocity或者LovelyDocs使用。
8. <kbd>1000-</kbd> [convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch): 将Torch t7模型转换为PyTorch模型。
9. <kbd>5000+</kbd> [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch): PyTorch 预训练卷积神经网络：NASNet, ResNeXt, ResNet, InceptionV4, InceptionResnetV2, Xception, DPN 等等。该项目的目标是帮助复制研究论文结果。
10. <kbd>1000-</kbd> [pytorch_fft](https://github.com/locuslab/pytorch_fft): CUDA FFTs的PyTorch包装器。
11. <kbd>1000-</kbd> [caffe_to_torch_to_pytorch](https://github.com/fanq15/caffe_to_torch_to_pytorch): Caffe模型转PyTorch/Torch模型，Torch模型转PyTorch模型。
12. <kbd>1000-</kbd> [pytorch-extension](https://github.com/sniklaus/pytorch-extension): PyTorch的CUDA扩展示例，计算了两个张量的[哈达玛积(Hadamard product)](https://baike.baidu.com/item/哈达玛积/18894493?fr=aladdin)。
13. <kbd>5500+</kbd> [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch): 该模块以tensorboard格式保存PyTorch张量以供检查。目前支持tensorboard中的标量、图像、音频、直方图等特性。
14. <kbd>1400+</kbd> [gpytorch](https://github.com/jrg365/gpytorch): GPyTorch是一个用PyTorch实现的高斯过程库。它可以轻松地创建可伸缩、灵活和模块化的高斯过程模型。
15. <kbd>1700+</kbd> [spotlight](https://github.com/maciejkula/spotlight): 深度推荐模型。
16. <kbd>1000-</kbd> [pytorch-cns](https://github.com/awentzonline/pytorch-cns): 基于PyTorch的广义压缩网络搜索（Generalized [Compressed Network Search](http://people.idsia.ch/~juergen/compressednetworksearch.html)）。
17. <kbd>1000-</kbd> [pyinn](https://github.com/szagoruyko/pyinn): CuPy实现融合PyTorch操作。
18. <kbd>1000-</kbd> [inferno](https://github.com/nasimrahaman/inferno): 关于PyTorch的实用程序库。
19. <kbd>1000-</kbd> [pytorch-fitmodule](https://github.com/henryre/pytorch-fitmodule): 一种用于PyTorch模块的超简单拟合方法。
20. <kbd>2400+</kbd> [inferno-sklearn](https://github.com/dnouri/inferno): 一个基于PyTorch封装且兼容scikit-learn的神经网络库。
21. <kbd>1000-</kbd> [pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert): 在 pytorch, caffe prototxt/weights 和 darknet cfg/weights 之间转换。
22. <kbd>1000-</kbd> [pytorch2caffe](https://github.com/longcw/pytorch2caffe): 将PyTorch模型转换成Caffe模型。
23. <kbd>1000-</kbd> [pytorch-tools](https://github.com/nearai/pytorch-tools): PyTorch工具。
24. <kbd>1700+</kbd> [sru](https://github.com/taolei87/sru): 训练RNNs和训练CNNs一样快。 (arxiv.org/abs/1709.02755)
25. <kbd>1000-</kbd> [torch2coreml](https://github.com/prisma-ai/torch2coreml): Torch7 -> CoreML，该工具可将Torch7模型转换为[Apple CoreML](https://developer.apple.com/documentation/coreml)格式以便在Apple设备上运行。
26. <kbd>1000-</kbd> [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding): PyTorch 深度纹理编码网络 (Deep Texture Encoding Network) http://hangzh.com/PyTorch-Encoding
27. <kbd>1000-</kbd> [pytorch-ctc](https://github.com/ryanleary/pytorch-ctc): PyTorch-CTC 实现了CTC(联结主义时间分类，Connectionist Temporal Classification)集束搜索（Beam Search）解码。C++代码借鉴了TensorFlow，并通过一些改进增加了灵活性。
28. <kbd>1000-</kbd> [candlegp](https://github.com/t-vi/candlegp): Pytorch中的高斯过程。
29. <kbd>1000-</kbd> [dpwa](https://github.com/loudinthecloud/dpwa): 基于成对平均（Pair-Wise Averaging）的分布式学习。
30. <kbd>1000-</kbd> [dni-pytorch](https://github.com/koz4k/dni-pytorch): 基于合成梯度的PyTorch解耦神经接口。
31. <kbd>2400+</kbd> [skorch](https://github.com/dnouri/skorch): 一个基于PyTorch封装且兼容scikit-learn的神经网络库。
32. <kbd>2100+</kbd> [ignite](https://github.com/pytorch/ignite): Ignite是一个高级库，帮助你在PyTorch中训练神经网络。
33. <kbd>1000-</kbd> [Arnold](https://github.com/glample/Arnold): Arnold - DOOM 游戏代理。
34. <kbd>1000-</kbd> [pytorch-mcn](https://github.com/albanie/pytorch-mcn): 将MatConvNet模型转换为PyTorch模型。
35. <kbd>1700+</kbd> [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch): Faster R-CNN 的简化实现，性能与原始论文相当。
36. <kbd>1000-</kbd> [generative_zoo](https://github.com/DL-IT/generative_zoo): generative_zoo提供了PyTorch中一些生成模型的工作实现。
37. <kbd>1000+</kbd> [pytorchviz](https://github.com/szagoruyko/pytorchviz): 可视化PyTorch的运行图。
38. <kbd>1000-</kbd> [cogitare](https://github.com/cogitare-ai/cogitare): Cogitare - 一个现代、快速、模块化的深度学习和机器学习框架。
39. <kbd>1000-</kbd> [pydlt](https://github.com/dmarnerides/pydlt): 基于PyTorch的深度学习工具箱。
40. <kbd>1000-</kbd> [semi-supervised-pytorch](https://github.com/wohlert/semi-supervised-pytorch): 各种基于VAE的半监督模型和生成模型的实现。
41. <kbd>1000-</kbd> [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster): 优化图簇算法的PyTorch扩展库。
42. <kbd>1000-</kbd> [neural-assembly-compiler](https://github.com/aditya-khant/neural-assembly-compiler): 基于自适应神经编译的PyTorch神经汇编编译器。
43. <kbd>1000-</kbd> [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch): 将Caffe模型转换为PyTorch模型。
44. <kbd>1000-</kbd> [extension-cpp](https://github.com/pytorch/extension-cpp): PyTorch中的C++扩展。
45. <kbd>1000-</kbd> [pytoune](https://github.com/GRAAL-Research/pytoune): 类Keras框架和实用程序。
46. <kbd>1000-</kbd> [jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement): 使用PyTorch，OpenAI Gym和Gazebo机器人模拟的NVIDIA Jetson深度强化学习GPU库。
47. <kbd>1000-</kbd> [matchbox](https://github.com/salesforce/matchbox): 编写单个示例的PyTorch代码，然后小批量地高效运行。
48. <kbd>1000-</kbd> [torch-two-sample](https://github.com/josipd/torch-two-sample): PyTorch双样本测试库。
49. <kbd>1500+</kbd> [pytorch-summary](https://github.com/sksq96/pytorch-summary): PyTorch模型总结，类似于Keras中的`model.summary()`。
50. <kbd>1000-</kbd> [mpl.pytorch](https://github.com/BelBES/mpl.pytorch): MaxPoolingLoss的PyTorch实现。
51. <kbd>null</kbd> [scVI-dev](https://github.com/YosefLab/scVI-dev): 链接失效。
52. <kbd>2300+</kbd> [apex](https://github.com/NVIDIA/apex): 一个PyTorch扩展：面向精简混合精度和分布式训练。
53. <kbd>2900+</kbd> [ELF](https://github.com/pytorch/ELF): ELF: 游戏研究平台，复现了AlphaGoZero/AlphaZero。
54. <kbd>1000-</kbd> [Torchlite](https://github.com/EKami/Torchlite): Pytorch建立在sklearn、Pytorch和Tensorflow等流行机器学习框架上的高水平库。
55. <kbd>1000-</kbd> [joint-vae](https://github.com/Schlumberger/joint-vae): JointVAE的PyTorch实现，一个面向分离连续和离散变异因素的框架 star2。
56. <kbd>1000-</kbd> [SLM-Lab](https://github.com/kengz/SLM-Lab): PyTorch模块化深度强化学习框架。
57. <kbd>1000-</kbd> [bindsnet](https://github.com/Hananel-Hazan/bindsnet): 一个Python包，可借助PyTorch `Tensor` 功能在CPUs或GPUs上模拟脉冲神经网络(SNNs, Spiking Neural Networks)。
58. <kbd>1000-</kbd> [pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch): 作为 PyTorch nn.Module 扩展的 ProGAN 包。
59. <kbd>4900+</kbd> [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric): PyTorch几何深度学习扩展库。
60. <kbd>1000-</kbd> [torchplus](https://github.com/knighton/torchplus): 在 PyTorch modules 上实现 + 运算符，返回序列。
61. <kbd>1000-</kbd> [lagom](https://github.com/zuoxingdong/lagom): lagom: 用于强化学习算法快速原型构建的轻量级PyTorch架构。
62. <kbd>1000-</kbd> [torchbearer](https://github.com/ecs-vlc/torchbearer): torchbearer: PyTorch模型拟合库。
63. <kbd>1000-</kbd> [pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl): 强化学习中的模型不可知元学习(MAML, Model-Agnostic Meta-Learning)。
64. <kbd>1000-</kbd> [NALU](https://github.com/bharathgs/NALU): 神经算术逻辑单元(Neural Arithmetic Logic Units)的PyTorch基本实现，论文：arxiv.org/pdf/1808.00508.pdf 。
65. <kbd>1000-</kbd> [QuCumber](https://github.com/PIQuIL/QuCumber): 神经网络多体波函数重构。
66. <kbd>1000-</kbd> [magnet](https://github.com/MagNet-DL/magnet): 自我建立的深度学习项目。http://magnet-dl.readthedocs.io/
67. <kbd>1000-</kbd> [opencv_transforms](https://github.com/jbohnslav/opencv_transforms): OpenCV实现Torchvision的图像分割。
68. <kbd>15300+</kbd> [fastai](https://github.com/fastai/fastai): fast.ai 深度学习库、课程和教程。
69. <kbd>1000-</kbd> [pytorch-dense-correspondence](https://github.com/RobotLocomotion/pytorch-dense-correspondence): [《Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation》](arxiv.org/pdf/1806.08756.pdf) 一文的代码。
70. <kbd>1000-</kbd> [colorization-pytorch](https://github.com/richzhang/colorization-pytorch): PyTorch实现交互式深度着色(Interactive Deep Colorization)。 richzhang.github.io/ideepcolor
71. <kbd>1000-</kbd> [beauty-net](https://github.com/cms-flash/beauty-net): PyTorch一个简单、灵活、可扩展的PyTorch模板。
72. <kbd>1000-</kbd> [OpenChem](https://github.com/Mariewelt/OpenChem): OpenChem: 面向计算化学和药物设计研究的深度学习工具包 mariewelt.github.io/OpenChem 。
73. <kbd>1000-</kbd> [torchani](https://github.com/aiqm/torchani): PyTorch精确神经网络电位。 aiqm.github.io/torchani
74. <kbd>1000-</kbd> [PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS): PyTorch实现L-BFGS。
75. <kbd>1400+</kbd> [gpytorch](https://github.com/cornellius-gp/gpytorch): PyTorch中对高斯过程的高效且模块化的实现。
76. <kbd>1000-</kbd> [hessian](https://github.com/mariogeiger/hessian): PyTorch版hessian。
77. <kbd>1000-</kbd> [vel](https://github.com/MillionIntegrals/vel): 深度学习研究中的速度。
78. <kbd>1000-</kbd> [nonechucks](https://github.com/msamogh/nonechucks): 动态地处理数据集中的坏样本，使用转换作为过滤器。
79. <kbd>1000-</kbd> [torchstat](https://github.com/Swall0w/torchstat): PyTorch中的模型分析器。
81. <kbd>1000+</kbd> [QNNPACK](https://github.com/pytorch/QNNPACK): Quantized Neural Network PACKage - mobile-optimized implementation of quantized neural network operators.
82. <kbd>2400+</kbd> [torchdiffeq](https://github.com/rtqichen/torchdiffeq): Differentiable ODE solvers with full GPU support and O(1)-memory backpropagation.
83. <kbd>1000-</kbd> [redner](https://github.com/BachiLi/redner): A differentiable Monte Carlo path tracer
84. <kbd>1000-</kbd> [pixyz](https://github.com/masa-su/pixyz): a library for developing deep generative models in a more concise, intuitive and extendable way.
85. <kbd>1000-</kbd> [euclidesdb](https://github.com/perone/euclidesdb): A multi-model machine learning feature embedding database http://euclidesdb.readthedocs.io
86. <kbd>1000-</kbd> [pytorch2keras](https://github.com/nerox8664/pytorch2keras): 将PyTorch模型转换为Keras模型。
87. <kbd>1000-</kbd> [salad](https://github.com/domainadaptation/salad): 域适应和半监督学习工具箱。
88. <kbd>1000-</kbd> [netharn](https://github.com/Erotemic/netharn): Parameterized fit and prediction harnesses for pytorch.
89. <kbd>2700+</kbd> [dgl](https://github.com/dmlc/dgl): Python package built to ease deep learning on graph, on top of existing DL frameworks. http://dgl.ai. 
90. <kbd>1300+</kbd> [gandissect](https://github.com/CSAILVision/gandissect): Pytorch-based tools for visualizing and understanding the neurons of a GAN. gandissect.csail.mit.edu 
91. <kbd>1000-</kbd> [delira](https://github.com/justusschock/delira): Lightweight framework for fast prototyping and training deep neural networks in medical imaging delira.rtfd.io
92. <kbd>1000-</kbd> [mushroom](https://github.com/AIRLab-POLIMI/mushroom): Python library for Reinforcement Learning experiments.
93. <kbd>1000-</kbd> [Xlearn](https://github.com/thuml/Xlearn): 迁移学习库。
94. <kbd>1000-</kbd> [geoopt](https://github.com/ferrine/geoopt): Riemannian Adaptive Optimization Methods with pytorch optim
95. <kbd>1000-</kbd> [vegans](https://github.com/unit8co/vegans): 包含多种现有的GANs。
96. <kbd>1000-</kbd> [torchgeometry](https://github.com/arraiyopensource/torchgeometry): TGM: PyTorch Geometry
97. <kbd>1000-</kbd> [AdverTorch](https://github.com/BorealisAI/advertorch): A Toolbox for Adversarial Robustness (attack/defense/training) Research
98. <kbd>2500+</kbd> [AdaBound](https://github.com/Luolc/AdaBound): An optimizer that trains as fast as Adam and as good as SGD.a
99. <kbd>1000-</kbd> [fenchel-young-losses](https://github.com/mblondel/fenchel-young-losses): Probabilistic classification in PyTorch/TensorFlow/scikit-learn with Fenchel-Young losses
100. <kbd>1000-</kbd> [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter): Count the FLOPs of your PyTorch model.
101. <kbd>1000-</kbd> [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
102. <kbd>1000-</kbd> [Catalyst](https://github.com/catalyst-team/catalyst): High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. Being able to research/develop something new, rather than write another regular train loop.
103. <kbd>1000-</kbd> [Ax](https://github.com/facebook/Ax): 自适应实验平台。
104. <kbd>1000-</kbd> [pywick](https://github.com/achaiah/pywick): High-level batteries-included neural network training library for Pytorch
105. <kbd>1000-</kbd> [torchgpipe](https://github.com/kakaobrain/torchgpipe): PyTorch实现GPipe。 torchgpipe.readthedocs.io
106. <kbd>1000-</kbd> [hub](https://github.com/pytorch/hub): Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.
107. <kbd>2000+</kbd> [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning): Rapid research framework for Pytorch. The researcher's version of keras.
108. <kbd>1000-</kbd> [Tor10](https://github.com/kaihsin/Tor10): A Generic Tensor-Network library that is designed for quantum simulation, base on the pytorch.
108. <kbd>2300+</kbd> [tensorwatch](https://github.com/microsoft/tensorwatch): Debugging, monitoring and visualization for Deep Learning and Reinforcement Learning from Microsoft Research.
109. <kbd>1000-</kbd> [wavetorch](https://github.com/fancompute/wavetorch): 波动方程的数值求解与反传播。 arxiv.org/abs/1904.12831
110. <kbd>1000-</kbd> [diffdist](https://github.com/ag14774/diffdist): diffdist is a python library for pytorch. It extends the default functionality of torch.autograd and adds support for differentiable communication between processes. 
111. <kbd>1000-</kbd> [torchprof](https://github.com/awwong1/torchprof): A minimal dependency library for layer-by-layer profiling of Pytorch models.
112. <kbd>1000-</kbd> [osqpth](https://github.com/oxfordcontrol/osqpth): The differentiable OSQP solver layer for PyTorch. 
113. <kbd>1000-</kbd> [mctorch](https://github.com/mctorch/mctorch): 面向深度学习的流形优化库。
114. <kbd>1000-</kbd> [pytorch-hessian-eigenthings](https://github.com/noahgolmant/pytorch-hessian-eigenthings): Efficient PyTorch Hessian eigendecomposition using the Hessian-vector product and stochastic power iteration. 
115. <kbd>1000-</kbd> [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine): Minkowski Engine is an auto-diff library for generalized sparse convolutions and high-dimensional sparse tensors.
116. <kbd>1000-</kbd> [pytorch-cpp-rl](https://github.com/Omegastick/pytorch-cpp-rl): CppRl是一个强化学习框架，用 PyTorch C++ 前端编写。
117. <kbd>1000-</kbd> [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt): PyTorch extensions for fast R&D prototyping and Kaggle farming
118. <kbd>1000-</kbd> [argus-tensor-stream](https://github.com/Fonbet/argus-tensor-stream): A library for real-time video stream decoding to CUDA memory tensorstream.argus-ai.com
129. <kbd>1000-</kbd> [macarico](https://github.com/hal3/macarico): 在PyTorch中学习搜索。

## Tutorials & examples｜教程 & 示例

1. <kbd>3500+</kbd> **[Practical Pytorch](https://github.com/spro/practical-pytorch)**: 该教程对不同的RNN模型进行了解释。
2. [DeepLearningForNLPInPytorch](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html): IPython Notebook 深度学习教程，包含对自然语言处理的强调。
3. <kbd>13100+</kbd> [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial): 面向研究人员的深度学习教程，其中大部分模型的实现代码都少于30行。
4. <kbd>1000-</kbd> [pytorch-exercises](https://github.com/keon/pytorch-exercises): PyTorch练习集合。
5. <kbd>2600+</kbd> [pytorch tutorials](https://github.com/pytorch/tutorials): 各种PyTorch教程。
6. <kbd>9900+</kbd> [pytorch examples](https://github.com/pytorch/examples):  PyTorch使用示例，应用场景包括视觉、文本、强化学习等。
7. <kbd>1000-</kbd> [pytorch practice](https://github.com/napsternxg/pytorch-practice): PyTorch示例。  
8. <kbd>1000-</kbd> [pytorch mini tutorials](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials): PyTorch极简教程，改编自Alec Radford的[Theano教程](https://github.com/Newmu/Theano-Tutorials)。
9. <kbd>1000-</kbd> [pytorch text classification](https://github.com/xiayandi/Pytorch_text_classification): PyTorch实现基于CNN的文本分类。
10. <kbd>1000-</kbd> [cats vs dogs](https://github.com/desimone/pytorch-cat-vs-dogs): Kaggle 竞赛 Dogs vs. Cats Redux: Kernels Edition 的网络微调示例。
11. <kbd>1000-</kbd> [convnet](https://github.com/eladhoffer/convNet.pytorch): 深度卷积网络在不同数据集(ImageNet, Cifar10, Cifar100, MNIST)上的完整训练示例。
12. <kbd>1000-</kbd> [pytorch-generative-adversarial-networks](https://github.com/mailmahee/pytorch-generative-adversarial-networks): 一个简单的对抗生成网络(GAN) 。
13. <kbd>1000-</kbd> [pytorch containers](https://github.com/amdegroot/pytorch-containers): PyTorch中简化的Torch容器。
14. <kbd>1000-</kbd> [T-SNE in pytorch](https://github.com/cemoody/topicsne): t-SNE实验。
15. <kbd>1000-</kbd> [AAE_pytorch](https://github.com/fducau/AAE_pytorch): PyTorch版对抗自编码器。
16. <kbd>1000-</kbd> [Kind_PyTorch_Tutorial](https://github.com/GunhoChoi/Kind_PyTorch_Tutorial): PyTorch新手教程。  
17. <kbd>1000-</kbd> [pytorch-poetry-gen](https://github.com/justdark/pytorch-poetry-gen): 基于PyTorch的char-RNN（字符级循环神经网络）。  
18. <kbd>1000-</kbd> [pytorch-REINFORCE](https://github.com/JamesChuanggg/pytorch-REINFORCE): PyTorch 实现了 OpenAI gym 下离散和连续控制的 REINFORCE。
19. <kbd>3600+</kbd> **[PyTorch-Tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial)**: 简单而快速地搭建你自己的神经网络。 https://morvanzhou.github.io/tutorials/
20. <kbd>1000-</kbd> [pytorch-intro](https://github.com/joansj/pytorch-intro): 演示如何在PyTorch中实现CNNs和RNNs。
21. <kbd>1000-</kbd> [pytorch-classification](https://github.com/bearpaw/pytorch-classification): 一个CIFAR-10/100和ImageNet数据集上的分类框架。
22. <kbd>1000-</kbd> [pytorch_notebooks - hardmaru](https://github.com/hardmaru/pytorch_notebooks): 用NumPy和PyTorch编写的随机教程。
23. <kbd>1000-</kbd> [pytorch_tutoria-quick](https://github.com/soravux/pytorch_tutorial): PyTorch介绍和教程。面向计算机视觉、图形和机器学习领域的研究人员，要求对神经网络理论知识和常用神经网络框架由基本的了解。
24. <kbd>1000-</kbd> [Pytorch_fine_tuning_Tutorial](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial): 在PyTorch中进行微调或转移学习的简短教程。
25. <kbd>1000-</kbd> [pytorch_exercises](https://github.com/Kyubyong/pytorch_exercises): PyTorch练习。
26. <kbd>1000-</kbd> [traffic-sign-detection](https://github.com/soumith/traffic-sign-detection-homework): 纽约大学2018年计算机视觉秋季课程示例。
27. <kbd>1000-</kbd> [mss_pytorch](https://github.com/Js-Mim/mss_pytorch): 无需进行滤波后处理，利用循环推断算法实现歌唱语音分离 - PyTorch 实现。 演示: js-mim.github.io/mss_pytorch
28. <kbd>2100+</kbd> [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch) cs-224n课程中的各种深度NLP模型的PyTorch实现。(Stanford Univ: NLP with Deep Learning)
29. <kbd>1000-</kbd> [Mila introductory tutorials](https://github.com/mila-udem/welcome_tutorials): 面向MILA新生的各种教程。（[MILA：加拿大蒙特利尔人工智能研究中心](https://mila.quebec/en/mila/)）
30. <kbd>1000-</kbd> [pytorch.rl.learning](https://github.com/moskomule/pytorch.rl.learning): 使用PyTorch学习强化学习。
31. <kbd>1000-</kbd> [minimal-seq2seq](https://github.com/keon/seq2seq): 关注神经机器翻译的最小Seq2Seq模型。
32. <kbd>1000-</kbd> [tensorly-notebooks](https://github.com/JeanKossaifi/tensorly-notebooks): 利用Python和TensorLy实现张量方法。 tensorly.github.io/dev
33. <kbd>1000-</kbd> [pytorch_bits](https://github.com/jpeg729/pytorch_bits): 时序预测的相关示例。
34. <kbd>1000-</kbd> [skip-thoughts](https://github.com/sanyam5/skip-thoughts): PyTorch实现Skip-Thought词向量模型。
35. <kbd>1000-</kbd> [video-caption-pytorch](https://github.com/xiadingZ/video-caption-pytorch): 利用PyTorch为视频添加字幕。
36. <kbd>1000-</kbd> [Capsule-Network-Tutorial](https://github.com/higgsfield/Capsule-Network-Tutorial): 简单易学的胶囊网络（Capsule Network）教程。
37. <kbd>1400+</kbd> [code-of-learn-deep-learning-with-pytorch](https://github.com/SherlockLiao/code-of-learn-deep-learning-with-pytorch): 《深度学习入门之PyTorch》书中代码。 item.jd.com/17915495606.html
38. <kbd>1600+</kbd> [RL-Adventure](https://github.com/higgsfield/RL-Adventure): Pytorch 版 Deep Q Learning 教程，简单、易学、代码可读性强，包含 DQN / DDQN / Prioritized replay/ noisy networks/ distributional values/ Rainbow/ hierarchical RL 的 PyTorch 实现。
39. <kbd>1000-</kbd> [accelerated_dl_pytorch](https://github.com/hpcgarage/accelerated_dl_pytorch): Jupyter Day Atlanta II 会议上的加速深度学习算法，包含 PyTorch 教程和会议演讲文稿。
40. <kbd>1800+</kbd> [RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2): 以下内容的 PyTorch0.4 版本教程: actor critic / proximal policy optimization / acer / ddpg / twin dueling ddpg / soft actor critic / generative adversarial imitation learning / hindsight experience replay。
41. [Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f): 50行生成对抗网络。
42. [adversarial-autoencoders-with-pytorch](https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/): PyTorch对抗自编码器。
43. [transfer learning using pytorch](https://medium.com/@vishnuvig/transfer-learning-using-pytorch-4c3475f4495): PyTorch迁移学习。
44. [how-to-implement-a-yolo-object-detector-in-pytorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/): 如何使用PyTorch实现一个YOLO (v3)物体检测器。
45. [pytorch-for-recommenders-101](http://blog.fastforwardlabs.com/2018/04/10/pytorch-for-recommenders-101.html): 使用PyTorch构建推荐系统。
46. <kbd>1000-</kbd> [pytorch-for-numpy-users](https://github.com/wkentaro/pytorch-for-numpy-users): 面向Numpy用户的PyTorch。
47. [PyTorch Tutorial](http://www.pytorchtutorial.com/): PyTorch中文教程（PyTorch中文网）。
48. <kbd>1000-</kbd> [grokking-pytorch](https://github.com/Kaixhin/grokking-pytorch): 手把手教你学会PyTorch。
49. <kbd>1100+</kbd> [PyTorch-Deep-Learning-Minicourse](https://github.com/Atcold/PyTorch-Deep-Learning-Minicourse): PyTorch深度学习微型课程。
50. <kbd>1000-</kbd> [pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples): PyTorch的一些自定义数据集示例。
51. [Multiplicative LSTM for sequence-based Recommenders](https://florianwilhelm.info/2018/08/multiplicative_LSTM_for_sequence_based_recos/): 面向基于序列的推荐器的乘法LSTM。/基于LSTM的序列推荐实现。
52. <kbd>1000-</kbd> [deeplearning.ai-pytorch](https://github.com/furkanu/deeplearning.ai-pytorch): Coursera深度学习课程(deeplearning.ai)任务的PyTorch实现。
53. <kbd>1000-</kbd> [MNIST_Pytorch_python_and_capi](https://github.com/tobiascz/MNIST_Pytorch_python_and_capi): 示例：如何在Python中训练一个MNIST网络并在C++中用PyTorch1.0运行。
54. <kbd>1000-</kbd> [torch_light](https://github.com/ne7ermore/torch_light): 教程和示例，包括强化学习、NLP、CV。Logistic、CNN、RNN、LSTM等神经网络模型由数行代码实现，一些高级示例由复杂模型实现。
55. <kbd>1000-</kbd> [portrain-gan](https://github.com/dribnet/portrain-gan): 编码（解码尚未实现）art-DCGAN 生成的肖像油画。
56. <kbd>1000-</kbd> [mri-analysis-pytorch](https://github.com/omarsar/mri-analysis-pytorch): 使用PyTorch和MedicalTorch进行核磁共振（MRI）分析。
57. <kbd>1000-</kbd> [cifar10-fast](https://github.com/davidcpage/cifar10-fast): 在79秒内完成CIFAR10数据集上的ResNet模型的训练并达到94%的测试准确率，相关内容参见 [blog series](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/)。
58. [Intro to Deep Learning with PyTorch](https://in.udacity.com/course/deep-learning-pytorch--ud188): Udacity和Facebook联合推出的免费课程，包括对PyTorch的介绍和对PyTorch作者之一的Soumith Chintala的采访。
59. <kbd>1000+</kbd> [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis): PyTorch和TorchText语义分析教程。
60. <kbd>1600+</kbd> [pytorch-image-models](https://github.com/rwightman/pytorch-image-models): PyTorch图像模型、脚本、与训练权重—— (SE)ResNet/ResNeXT, DPN, EfficientNet, MobileNet-V3/V2/V1, MNASNet, Single-Path NAS, FBNet等等。
61. <kbd>1000-</kbd> [CIFAR-ZOO](https://github.com/BIGBALLON/CIFAR-ZOO): 以CIFAR为基准的多种CNN架构的PyTorch实现。
62. <kbd>1000-</kbd> [d2l-pytorch](https://github.com/dsgiitr/d2l-pytorch): 本项目尝试复制《动手深度学习（Dive into Deep Learning）》(www.d2l.ai)一书，将MXnet代码改编为PyTorch版。
63. <kbd>1000-</kbd> [thinking-in-tensors-writing-in-pytorch](https://github.com/stared/thinking-in-tensors-writing-in-pytorch):  张量思维，PyTorch实践 (深度学习入门)。
64. <kbd>1000-</kbd> [NER-BERT-pytorch](https://github.com/lemonhu/NER-BERT-pytorch): 命名试题识别的PyTorch解决方案，使用了Google AI的预训练BERT模型。
65. <kbd>1000-</kbd> [pytorch-sync-batchnorm-example](https://github.com/dougsouza/pytorch-sync-batchnorm-example): 如何在 PyTorch 中使用交叉复制（Cross Replica）/同步批标准化（Synchronized Batchnorm）。

## Paper implementations｜论文实现

1. <kbd>1000-</kbd> [google_evolution](https://github.com/neuralix/google_evolution): 实现了 [Large-scale evolution of image classifiers](https://arxiv.org/abs/1703.01041) 一文的结果网络之一。
2. <kbd>1000-</kbd> [pyscatwave](https://github.com/edouardoyallon/pyscatwave): 基于CuPy/PyTorch的快速散射变换，[Scaling the Scattering Transform: Deep Hybrid Networks](https://arxiv.org/abs/1703.08961)
3. <kbd>1000-</kbd> [scalingscattering](https://github.com/edouardoyallon/scalingscattering): 该仓库包含 [Scaling The Scattering Transform : Deep Hybrid Networks](https://arxiv.org/abs/1703.08961) 一文中的实验。  
4. <kbd>1000-</kbd> [deep-auto-punctuation](https://github.com/episodeyang/deep-auto-punctuation): 通过逐字符学习实现自动添加标点。
5. <kbd>1000-</kbd> [Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation): 基于PyTorch的多人人体姿态估计，[原始代码](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)。
6. <kbd>1000-</kbd> [PyTorch-value-iteration-networks](https://github.com/onlytailei/PyTorch-value-iteration-networks): PyTorch实现价值迭代网络（[Value Iteration Networks](https://arxiv.org/abs/1602.02867)）（NIPS2016最佳论文奖）。
7. <kbd>1000-</kbd> [pytorch_Highway](https://github.com/analvikingur/pytorch_Highway): PyTorch实现高速公路网络（[Highway Networks](https://arxiv.org/abs/1505.00387)）。
8. <kbd>1000-</kbd> [pytorch_NEG_loss](https://github.com/analvikingur/pytorch_NEG_loss): PyTorch实现负采样损失（[Negative Sampling Loss](https://arxiv.org/abs/1310.4546)）。  
9. <kbd>1000-</kbd> [pytorch_RVAE](https://github.com/analvikingur/pytorch_RVAE): 用PyTorch实现的产生序列数据的递归变分自动编码器，相关论文：[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#)，[Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)。
10. <kbd>1000-</kbd> [pytorch_TDNN](https://github.com/analvikingur/pytorch_TDNN): 用PyTorch实现时间延迟神经网络（Time Delayed NN）。
11. <kbd>1000-</kbd> [eve.pytorch](https://github.com/moskomule/eve.pytorch): 一个Eve优化器的实现，相关论文：[Imploving Stochastic Gradient Descent with Feedback](https://arxiv.org/abs/1611.01505)。  
12. <kbd>1000-</kbd> [e2e-model-learning](https://github.com/locuslab/e2e-model-learning): 随机优化中的基于任务的端到端模型，https://arxiv.org/abs/1703.04529 。
13. <kbd>1000-</kbd> [pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch): PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".   
14. <kbd>2700+</kbd> [Single Shot MultiBox Detector](https://github.com/amdegroot/ssd.pytorch): A PyTorch Implementation of Single Shot MultiBox Detector.  
15. <kbd>1000-</kbd> [DiscoGAN](https://github.com/carpedm20/DiscoGAN-pytorch): PyTorch implementation of "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks"  
16. <kbd>1000-</kbd> [official DiscoGAN implementation](https://github.com/SKTBrain/DiscoGAN): Official implementation of "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks".  
17. <kbd>1000-</kbd> [pytorch-es](https://github.com/atgambardella/pytorch-es): This is a PyTorch implementation of [Evolution Strategies](https://arxiv.org/abs/1703.03864) .  
18. <kbd>1000-</kbd> [piwise](https://github.com/bodokaiser/piwise): Pixel-wise segmentation on VOC2012 dataset using pytorch.  
19. <kbd>1000-</kbd> [pytorch-dqn](https://github.com/transedward/pytorch-dqn): Deep Q-Learning Network in pytorch.  
20. <kbd>1000-</kbd> [neuraltalk2-pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch): image captioning model in pytorch(finetunable cnn in branch with_finetune)
21. <kbd>1000-</kbd> [vnet.pytorch](https://github.com/mattmacy/vnet.pytorch): A Pytorch implementation for V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.    
22. <kbd>1000-</kbd> [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn): PyTorch implementation of Fully Convolutional Networks.  
23. <kbd>1000-</kbd> [WideResNets](https://github.com/xternalz/WideResNet-pytorch): WideResNets for CIFAR10/100 implemented in PyTorch. This implementation requires less GPU memory than what is required by the official Torch implementation: https://github.com/szagoruyko/wide-residual-networks .
24. <kbd>1000-</kbd> [pytorch_highway_networks](https://github.com/c0nn3r/pytorch_highway_networks): Highway networks implemented in PyTorch.  
25. <kbd>1000-</kbd> [pytorch-NeuCom](https://github.com/ypxie/pytorch-NeuCom): Pytorch implementation of DeepMind's differentiable neural computer paper.  
26. <kbd>1000-</kbd> [captionGen](https://github.com/eladhoffer/captionGen): Generate captions for an image using PyTorch.  
27. <kbd>1000-</kbd> [AnimeGAN](https://github.com/jayleicn/animeGAN): A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing. 
28. <kbd>1000-</kbd> [Cnn-text classification](https://github.com/Shawn1993/cnn-text-classification-pytorch): This is the implementation of Kim's Convolutional Neural Networks for Sentence Classification paper in PyTorch.  
29. <kbd>1000+</kbd> [deepspeech2](https://github.com/SeanNaren/deepspeech.pytorch): Implementation of DeepSpeech2 using Baidu Warp-CTC. Creates a network based on the DeepSpeech2 architecture, trained with the CTC activation function.
30. <kbd>1000-</kbd> [seq2seq](https://github.com/MaximumEntropy/Seq2Seq-PyTorch): This repository contains implementations of Sequence to Sequence (Seq2Seq) models in PyTorch  
31. <kbd>1000-</kbd> [Asynchronous Advantage Actor-Critic in PyTorch](https://github.com/rarilurelo/pytorch_a3c): This is PyTorch implementation of A3C as described in Asynchronous Methods for Deep Reinforcement Learning. Since PyTorch has a easy method to control shared memory within multiprocess, we can easily implement asynchronous method like A3C.    
32. <kbd>1000-</kbd> [densenet](https://github.com/bamos/densenet.pytorch): This is a PyTorch implementation of the DenseNet-BC architecture as described in the paper Densely Connected Convolutional Networks by G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. This implementation gets a CIFAR-10+ error rate of 4.77 with a 100-layer DenseNet-BC with a growth rate of 12. Their official implementation and links to many other third-party implementations are available in the liuzhuang13/DenseNet repo on GitHub.  
33. <kbd>1000-</kbd> [nninit](https://github.com/alykhantejani/nninit): Weight initialization schemes for PyTorch nn.Modules. This is a port of the popular nninit for Torch7 by @kaixhin.  
34. <kbd>1200+</kbd> [faster rcnn](https://github.com/longcw/faster_rcnn_pytorch): This is a PyTorch implementation of Faster RCNN. This project is mainly based on py-faster-rcnn and TFFRCNN.For details about R-CNN please refer to the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 
35. <kbd>1000-</kbd> [doomnet](https://github.com/akolishchak/doom-net-pytorch): PyTorch's version of Doom-net implementing some RL models in ViZDoom environment.  
36. <kbd>1000-</kbd> [flownet](https://github.com/ClementPinard/FlowNetPytorch): Pytorch implementation of FlowNet by Dosovitskiy et al.  
37. <kbd>1000-</kbd> [sqeezenet](https://github.com/gsp-27/pytorch_Squeezenet): Implementation of Squeezenet in pytorch, #### pretrained models on CIFAR10 data to come Plan to train the model on cifar 10 and add block connections too.  
38. <kbd>2400+</kbd> [WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN): wassersteinGAN in pytorch. 
39. <kbd>1000-</kbd> [optnet](https://github.com/locuslab/optnet): This repository is by Brandon Amos and J. Zico Kolter and contains the PyTorch source code to reproduce the experiments in our paper OptNet: Differentiable Optimization as a Layer in Neural Networks.  
40. <kbd>1000-</kbd> [qp solver](https://github.com/locuslab/qpth): A fast and differentiable QP solver for PyTorch. Crafted by Brandon Amos and J. Zico Kolter.  
41. <kbd>1000-</kbd> [Continuous Deep Q-Learning with Model-based Acceleration ](https://github.com/ikostrikov/pytorch-naf): Reimplementation of Continuous Deep Q-Learning with Model-based Acceleration.  
42. <kbd>1000-</kbd> [Learning to learn by gradient descent by gradient descent](https://github.com/ikostrikov/pytorch-meta-optimizer): PyTorch implementation of Learning to learn by gradient descent by gradient descent.
43. <kbd>1000-</kbd> [fast-neural-style](https://github.com/darkstar112358/fast-neural-style): pytorch implementation of fast-neural-style, The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with Instance Normalization.
44. <kbd>1000-</kbd> [PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer): Implementation of Neural Style Transfer in Pytorch. 
45. <kbd>1000-</kbd> [Fast Neural Style for Image Style Transform by Pytorch](https://github.com/bengxy/FastNeuralStyle): Fast Neural Style for Image Style Transform by Pytorch .
46. <kbd>1000-</kbd> [neural style transfer](https://github.com/alexis-jacq/Pytorch-Tutorials): An introduction to PyTorch through the Neural-Style algorithm (https://arxiv.org/abs/1508.06576) developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.   
47. <kbd>1000-</kbd> [VIN_PyTorch_Visdom](https://github.com/zuoxingdong/VIN_PyTorch_Visdom): PyTorch implementation of Value Iteration Networks (VIN): Clean, Simple and Modular. Visualization in Visdom.  
48. <kbd>1200+</kbd> [YOLO2](https://github.com/longcw/yolo2-pytorch): YOLOv2 in PyTorch.   
49. <kbd>1000-</kbd> [attention-transfer](https://github.com/szagoruyko/attention-transfer): Attention transfer in pytorch, read the paper [here](https://arxiv.org/abs/1612.03928).  
50. <kbd>1000-</kbd> [SVHNClassifier](https://github.com/potterhsu/SVHNClassifier-PyTorch): A PyTorch implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf).  
51. <kbd>1000-</kbd> [pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv): PyTorch implementation of Deformable Convolution.  
52. <kbd>1000-</kbd> [BEGAN-pytorch](https://github.com/carpedm20/BEGAN-pytorch): PyTorch implementation of [BEGAN](https://arxiv.org/abs/1703.10717): Boundary Equilibrium Generative Adversarial Networks.  
53. <kbd>1000-</kbd> [treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch): Tree LSTM implementation in PyTorch.
54. <kbd>1000-</kbd> [AGE](https://github.com/DmitryUlyanov/AGE): Code for paper "Adversarial Generator-Encoder Networks" by Dmitry Ulyanov, Andrea Vedaldi and Victor Lempitsky which can be found [here](http://sites.skoltech.ru/app/data/uploads/sites/25/2017/04/AGE.pdf) 
55. <kbd>1000-</kbd> [ResNeXt.pytorch](https://github.com/prlz77/ResNeXt.pytorch): Reproduces ResNet-V3 (Aggregated Residual Transformations for Deep Neural Networks) with pytorch.
56. <kbd>1000-</kbd> [pytorch-rl](https://github.com/jingweiz/pytorch-rl): Deep Reinforcement Learning with pytorch & visdom  
57. <kbd>1000-</kbd> [Deep-Leafsnap](https://github.com/sujithv28/Deep-Leafsnap): LeafSnap replicated using deep neural networks to test accuracy compared to traditional computer vision methods.  
58. <kbd>9400+</kbd> [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): PyTorch implementation for both unpaired and paired image-to-image translation.
59. <kbd>1000-</kbd> [A3C-PyTorch](https://github.com/onlytailei/A3C-PyTorch):PyTorch implementation of Advantage async actor-critic Algorithms (A3C) in PyTorch
60. <kbd>1000-</kbd> [pytorch-value-iteration-networks](https://github.com/kentsommer/pytorch-value-iteration-networks): Pytorch implementation of Value Iteration Networks (NIPS 2016 best paper)  
61. <kbd>1000-</kbd> [PyTorch-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Style-Transfer): PyTorch Implementation of Multi-style Generative Network for Real-time Transfer
62. <kbd>1000-</kbd> [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet): pytorch-deeplab-resnet-model.
63. <kbd>1000-</kbd> [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch): pytorch implementation for "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" https://arxiv.org/abs/1612.00593  
64. <kbd>1600+</kbd> **[pytorch-playground](https://github.com/aaron-xichen/pytorch-playground): Base pretrained models and datasets in pytorch (MNIST, SVHN, CIFAR10, CIFAR100, STL10, AlexNet, VGG16, VGG19, ResNet, Inception, SqueezeNet)**.
65. <kbd>1000-</kbd> [pytorch-dnc](https://github.com/jingweiz/pytorch-dnc): Neural Turing Machine (NTM) & Differentiable Neural Computer (DNC) with pytorch & visdom. 
66. <kbd>1000-</kbd> [pytorch_image_classifier](https://github.com/jinfagang/pytorch_image_classifier): Minimal But Practical Image Classifier Pipline Using Pytorch, Finetune on ResNet18, Got 99% Accuracy on Own Small Datasets.  
67. <kbd>1000-</kbd> [mnist-svhn-transfer](https://github.com/yunjey/mnist-svhn-transfer): PyTorch Implementation of CycleGAN and SGAN for Domain Transfer (Minimal).
68. <kbd>1000-</kbd> [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2): pytorch-yolo2
69. <kbd>1000-</kbd> [dni](https://github.com/andrewliao11/dni.pytorch): Implement Decoupled Neural Interfaces using Synthetic Gradients in Pytorch
70. <kbd>1000-</kbd> [wgan-gp](https://github.com/caogang/wgan-gp): A pytorch implementation of Paper "Improved Training of Wasserstein GANs".
71. <kbd>1000-</kbd> [pytorch-seq2seq-intent-parsing](https://github.com/spro/pytorch-seq2seq-intent-parsing): Intent parsing and slot filling in PyTorch with seq2seq + attention
72. <kbd>1000-</kbd> [pyTorch_NCE](https://github.com/demelin/pyTorch_NCE): An implementation of the Noise Contrastive Estimation algorithm for pyTorch. Working, yet not very efficient.
73. <kbd>1000-</kbd> [molencoder](https://github.com/cxhernandez/molencoder): Molecular AutoEncoder in PyTorch
74. <kbd>1000-</kbd> [GAN-weight-norm](https://github.com/stormraiser/GAN-weight-norm): Code for "On the Effects of Batch and Weight Normalization in Generative Adversarial Networks"
75. <kbd>1000-</kbd> [lgamma](https://github.com/rachtsingh/lgamma): Implementations of polygamma, lgamma, and beta functions for PyTorch
76. <kbd>1000-</kbd> [bigBatch](https://github.com/eladhoffer/bigBatch): Code used to generate the results appearing in "Train longer, generalize better: closing the generalization gap in large batch training of neural networks" 
77. <kbd>1000-</kbd> [rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch): Reinforcement learning with implementation of A3C LSTM for Atari 2600. 
78. <kbd>1000-</kbd> [pytorch-retraining](https://github.com/ahirner/pytorch-retraining): Transfer Learning Shootout for PyTorch's model zoo (torchvision)
79. <kbd>1000-</kbd> [nmp_qc](https://github.com/priba/nmp_qc): Neural Message Passing for Computer Vision
80. <kbd>1000-</kbd> [grad-cam](https://github.com/jacobgil/pytorch-grad-cam): Pytorch implementation of Grad-CAM
81. <kbd>1000-</kbd> [pytorch-trpo](https://github.com/mjacar/pytorch-trpo): PyTorch Implementation of Trust Region Policy Optimization (TRPO)
82. <kbd>1000-</kbd> [pytorch-explain-black-box](https://github.com/jacobgil/pytorch-explain-black-box): PyTorch implementation of Interpretable Explanations of Black Boxes by Meaningful Perturbation
83. <kbd>1000-</kbd> [vae_vpflows](https://github.com/jmtomczak/vae_vpflows): Code in PyTorch for the convex combination linear IAF and the Householder Flow, J.M. Tomczak & M. Welling https://jmtomczak.github.io/deebmed.html 
84. <kbd>1000-</kbd> [relational-networks](https://github.com/kimhc6028/relational-networks): Pytorch implementation of "A simple neural network module for relational reasoning" (Relational Networks) https://arxiv.org/pdf/1706.01427.pdf
85. <kbd>1000-</kbd> [vqa.pytorch](https://github.com/Cadene/vqa.pytorch): Visual Question Answering in Pytorch
86. <kbd>1200+</kbd> [end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator): Deal or No Deal? End-to-End Learning for Negotiation Dialogues
87. <kbd>1000-</kbd> [odin-pytorch](https://github.com/ShiyuLiang/odin-pytorch): Principled Detection of Out-of-Distribution Examples in Neural Networks. 
88. <kbd>1000-</kbd> [FreezeOut](https://github.com/ajbrock/FreezeOut): Accelerate Neural Net Training by Progressively Freezing Layers. 
89. <kbd>1000-</kbd> [ARAE](https://github.com/jakezhaojb/ARAE): Code for the paper "Adversarially Regularized Autoencoders for Generating Discrete Structures" by Zhao, Kim, Zhang, Rush and LeCun.
90. <kbd>1000-</kbd> [forward-thinking-pytorch](https://github.com/kimhc6028/forward-thinking-pytorch): Pytorch implementation of "Forward Thinking: Building and Training Neural Networks One Layer at a Time" https://arxiv.org/pdf/1706.02480.pdf  
91. <kbd>1000-</kbd> [context_encoder_pytorch](https://github.com/BoyuanJiang/context_encoder_pytorch): PyTorch Implement of Context Encoders
92. <kbd>3000+</kbd> [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch): A PyTorch implementation of the Transformer model in "Attention is All You Need".https://github.com/thnkim/OpenFacePytorch
93. <kbd>1000-</kbd> [OpenFacePytorch](https://github.com/thnkim/OpenFacePytorch): PyTorch module to use OpenFace's nn4.small2.v1.t7 model 
94. <kbd>1000-</kbd> [neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch):  PyTorch implementation of Neural Combinatorial Optimization with Reinforcement Learning.
95. <kbd>1000-</kbd> [pytorch-nec](https://github.com/mjacar/pytorch-nec): PyTorch Implementation of Neural Episodic Control (NEC)
96. <kbd>1000-</kbd> [seq2seq.pytorch](https://github.com/eladhoffer/seq2seq.pytorch): Sequence-to-Sequence learning using PyTorch
97. <kbd>1000-</kbd> [Pytorch-Sketch-RNN](https://github.com/alexis-jacq/Pytorch-Sketch-RNN): a pytorch implementation of arxiv.org/abs/1704.03477
98. <kbd>1000-</kbd> [pytorch-pruning](https://github.com/jacobgil/pytorch-pruning): PyTorch Implementation of [1611.06440] Pruning Convolutional Neural Networks for Resource Efficient Inference
99. <kbd>1000-</kbd> [DrQA](https://github.com/hitvoice/DrQA): A pytorch implementation of Reading Wikipedia to Answer Open-Domain Questions.
100. <kbd>1000-</kbd> [YellowFin_Pytorch](https://github.com/JianGoForIt/YellowFin_Pytorch): auto-tuning momentum SGD optimizer
101. <kbd>1000-</kbd> [samplernn-pytorch](https://github.com/deepsound-project/samplernn-pytorch): PyTorch implementation of SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. 
102. <kbd>1000-</kbd> [AEGeAN](https://github.com/tymokvo/AEGeAN): Deeper DCGAN with AE stabilization
103. <kbd>1000-</kbd> [/pytorch-SRResNet](https://github.com/twtygqyy/pytorch-SRResNet): pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network arXiv:1609.04802v2 
104. <kbd>1000-</kbd> [vsepp](https://github.com/fartashf/vsepp): Code for the paper "VSE++: Improved Visual Semantic Embeddings"
105. <kbd>1000-</kbd> [Pytorch-DPPO](https://github.com/alexis-jacq/Pytorch-DPPO): Pytorch implementation of Distributed Proximal Policy Optimization: arxiv.org/abs/1707.02286
106. <kbd>1400+</kbd> [UNIT](https://github.com/mingyuliutw/UNIT): PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation
107. <kbd>1000+</kbd> [efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch): A memory-efficient implementation of DenseNets
108. <kbd>1000-</kbd> [tsn-pytorch](https://github.com/yjxiong/tsn-pytorch): Temporal Segment Networks (TSN) in PyTorch.
109. <kbd>1000-</kbd> [SMASH](https://github.com/ajbrock/SMASH): An experimental technique for efficiently exploring neural architectures.
110. <kbd>1000-</kbd> [pytorch-retinanet](https://github.com/kuangliu/pytorch-retinanet): RetinaNet in PyTorch
111. <kbd>1000-</kbd> [biogans](https://github.com/aosokin/biogans):  Implementation supporting the ICCV 2017 paper "GANs for Biological Image Synthesis". 
112. <kbd>null</kbd> [Semantic Image Synthesis via Adversarial Learning]( https://github.com/woozzu/dong_iccv_2017): A PyTorch implementation of the paper "Semantic Image Synthesis via Adversarial Learning" in ICCV 2017. 
113. <kbd>1000-</kbd> [fmpytorch](https://github.com/jmhessel/fmpytorch): A PyTorch implementation of a Factorization Machine module in cython.
114. <kbd>1000-</kbd> [ORN](https://github.com/ZhouYanzhao/ORN): A PyTorch implementation of the paper "Oriented Response Networks" in CVPR 2017. 
115. <kbd>1000-</kbd> [pytorch-maml](https://github.com/katerakelly/pytorch-maml): PyTorch implementation of MAML: arxiv.org/abs/1703.03400
116. <kbd>1700+</kbd> [pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections):  Collection of generative models in Pytorch version.
117. <kbd>1000-</kbd> [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017): Pytorch Implementation of winner from VQA Chllange Workshop in CVPR'17. 
118. <kbd>1000-</kbd> [tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch):  PyTorch implementation of Tacotron speech synthesis model. 
119. <kbd>1000-</kbd> [pspnet-pytorch](https://github.com/Lextal/pspnet-pytorch): PyTorch implementation of PSPNet segmentation network
120. <kbd>1000-</kbd> [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF): Empower Sequence Labeling with Task-Aware Language Model http://arxiv.org/abs/1709.04109
121. <kbd>3100+</kbd> [face-alignment](https://github.com/1adrianb/face-alignment): Pytorch implementation of the paper "How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)", ICCV 2017
122. <kbd>1000-</kbd> [DepthNet](https://github.com/ClementPinard/DepthNet): PyTorch DepthNet Training on Still Box dataset. 
123. <kbd>1000-</kbd> [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch): PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution' (CVPRW 2017)
124. <kbd>1000-</kbd> [e2c-pytorch](https://github.com/ethanluoyc/e2c-pytorch): Embed to Control implementation in PyTorch.
125. <kbd>1500+</kbd> [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch): 3D ResNets for Action Recognition.
126. <kbd>1000-</kbd> [bandit-nmt](https://github.com/khanhptnk/bandit-nmt): This is code repo for our EMNLP 2017 paper "Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback", which implements the A2C algorithm on top of a neural encoder-decoder model and benchmarks the combination under simulated noisy rewards.
127. <kbd>1500+</kbd> [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr): PyTorch implementation of Advantage Actor Critic (A2C), Proximal Policy Optimization (PPO) and Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation (ACKTR).
128. <kbd>1000-</kbd> [zalando-pytorch](https://github.com/baldassarreFe/zalando-pytorch): Various experiments on the [Fashion-MNIST](zalandoresearch/fashion-mnist) dataset from Zalando.
129. <kbd>1000-</kbd> [sphereface_pytorch](https://github.com/clcarwin/sphereface_pytorch): A PyTorch Implementation of SphereFace.
130. <kbd>1000-</kbd> [Categorical DQN](https://github.com/floringogianu/categorical-dqn): A PyTorch Implementation of Categorical DQN from [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887).
131. <kbd>1000-</kbd> [pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm): pytorch ntm implementation. 
132. <kbd>null</kbd> [mask_rcnn_pytorch](https://github.com/felixgwu/mask_rcnn_pytorch): Mask RCNN in PyTorch.
133. <kbd>1000-</kbd> [graph_convnets_pytorch](https://github.com/xbresson/graph_convnets_pytorch): PyTorch implementation of graph ConvNets, NIPS’16
134. <kbd>1300+</kbd> [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn): A pytorch implementation of faster RCNN detection framework based on Xinlei Chen's tf-faster-rcnn.
135. <kbd>1000-</kbd> [torchMoji](https://github.com/huggingface/torchMoji): A pyTorch implementation of the DeepMoji model: state-of-the-art deep learning model for analyzing sentiment, emotion, sarcasm etc.
136. <kbd>2500+</kbd> [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch): Pytorch implementation for Semantic Segmentation/Scene Parsing on [MIT ADE20K dataset](http://sceneparsing.csail.mit.edu)
137. <kbd>1000+</kbd> [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn): PyTorch implementation of the Quasi-Recurrent Neural Network - up to 16 times faster than NVIDIA's cuDNN LSTM
138. <kbd>1000-</kbd> [pytorch-sgns](https://github.com/theeluwin/pytorch-sgns): Skipgram Negative Sampling in PyTorch.
139. <kbd>1000-</kbd> [SfmLearner-Pytorch ](https://github.com/ClementPinard/SfmLearner-Pytorch): Pytorch version of SfmLearner from Tinghui Zhou et al.
140. <kbd>1000-</kbd> [deformable-convolution-pytorch](https://github.com/1zb/deformable-convolution-pytorch): PyTorch implementation of Deformable Convolution. 
141. <kbd>1000-</kbd> [skip-gram-pytorch](https://github.com/fanglanting/skip-gram-pytorch): A complete pytorch implementation of skipgram model (with subsampling and negative sampling). The embedding result is tested with Spearman's rank correlation.
142. <kbd>1000-</kbd> [stackGAN-v2](https://github.com/hanzhanggit/StackGAN-v2): Pytorch implementation for reproducing StackGAN_v2 results in the paper StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks by Han Zhang*, Tao Xu*, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas.
143. <kbd>1000-</kbd> [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch): Unofficial pytorch implementation for Self-critical Sequence Training for Image Captioning. 
144. <kbd>1700+</kbd> [pygcn](https://github.com/tkipf/pygcn): Graph Convolutional Networks in PyTorch.
145. <kbd>1000-</kbd> [dnc](https://github.com/ixaxaar/pytorch-dnc): Differentiable Neural Computers, for Pytorch
146. <kbd>1000-</kbd> [prog_gans_pytorch_inference](https://github.com/ptrblck/prog_gans_pytorch_inference): PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot.
147. <kbd>1000-</kbd> [pytorch-capsule](https://github.com/timomernick/pytorch-capsule): Pytorch implementation of Hinton's Dynamic Routing Between Capsules.
148. <kbd>1000-</kbd> [PyramidNet-PyTorch](https://github.com/dyhan0920/PyramidNet-PyTorch): A PyTorch implementation for PyramidNets (Deep Pyramidal Residual Networks, arxiv.org/abs/1610.02915)
149. <kbd>1000-</kbd> [radio-transformer-networks](https://github.com/gram-ai/radio-transformer-networks): A PyTorch implementation of Radio Transformer Networks from the paper "An Introduction to Deep Learning for the Physical Layer". arxiv.org/abs/1702.00832
150. <kbd>1000-</kbd> [honk](https://github.com/castorini/honk): PyTorch reimplementation of Google's TensorFlow CNNs for keyword spotting.
151. <kbd>1000-</kbd> [DeepCORAL](https://github.com/SSARCandy/DeepCORAL): A PyTorch implementation of 'Deep CORAL: Correlation Alignment for Deep Domain Adaptation.', ECCV 2016
152. <kbd>1000-</kbd> [pytorch-pose](https://github.com/bearpaw/pytorch-pose): A PyTorch toolkit for 2D Human Pose Estimation.
153. <kbd>1000-</kbd> [lang-emerge-parlai](https://github.com/karandesai-96/lang-emerge-parlai): Implementation of EMNLP 2017 Paper "Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog" using PyTorch and ParlAI
154. <kbd>1000-</kbd> [Rainbow](https://github.com/Kaixhin/Rainbow): Rainbow: Combining Improvements in Deep Reinforcement Learning 
155. <kbd>1000-</kbd> [pytorch_compact_bilinear_pooling v1](https://github.com/gdlg/pytorch_compact_bilinear_pooling): This repository has a pure Python implementation of Compact Bilinear Pooling and Count Sketch for PyTorch.
156. <kbd>1000-</kbd> [CompactBilinearPooling-Pytorch v2](https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch): (Yang Gao, et al.) A Pytorch Implementation for Compact Bilinear Pooling.
157. <kbd>1000-</kbd> [FewShotLearning](https://github.com/gitabcworld/FewShotLearning): Pytorch implementation of the paper "Optimization as a Model for Few-Shot Learning"
158. <kbd>1000-</kbd> [meProp](https://github.com/jklj077/meProp): Codes for "meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting".
159. <kbd>1000-</kbd> [SFD_pytorch](https://github.com/clcarwin/SFD_pytorch): A PyTorch Implementation of Single Shot Scale-invariant Face Detector.
160. <kbd>1000-</kbd> [GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory): Continuum Learning with GEM: Gradient Episodic Memory. https://arxiv.org/abs/1706.08840
161. <kbd>1400+</kbd> [DeblurGAN](https://github.com/KupynOrest/DeblurGAN): Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks.
162. <kbd>4000+</kbd> [StarGAN](https://github.com/yunjey/StarGAN): StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Tranlsation.
163. <kbd>1000-</kbd> [CapsNet-pytorch](https://github.com/adambielski/CapsNet-pytorch): PyTorch implementation of NIPS 2017 paper Dynamic Routing Between Capsules.
164. <kbd>1000-</kbd> [CondenseNet](https://github.com/ShichenLiu/CondenseNet): CondenseNet: An Efficient DenseNet using Learned Group Convolutions.
165. <kbd>5000+</kbd> [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior): Image restoration with neural networks but without learning.
166. <kbd>1000-</kbd> [deep-head-pose](https://github.com/natanielruiz/deep-head-pose): Deep Learning Head Pose Estimation using PyTorch.
167. <kbd>1000-</kbd> [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing): This code has the source code for the paper "Random Erasing Data Augmentation".
168. <kbd>1000-</kbd> [FaderNetworks](https://github.com/facebookresearch/FaderNetworks): Fader Networks: Manipulating Images by Sliding Attributes - NIPS 2017
169. <kbd>1400+</kbd> [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch): FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
170. <kbd>3800+</kbd> [pix2pixHD](https://github.com/NVIDIA/pix2pixHD): Synthesizing and manipulating 2048x1024 images with conditional GANs tcwang0509.github.io/pix2pixHD 
171. <kbd>1000-</kbd> [pytorch-smoothgrad](https://github.com/pkdn/pytorch-smoothgrad): SmoothGrad implementation in PyTorch
172. <kbd>1000-</kbd> [RetinaNet](https://github.com/c0nn3r/RetinaNet): An implementation of RetinaNet in PyTorch.
173. <kbd>3900+</kbd> [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch): This project is a faster faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models. 
174. <kbd>1000-</kbd> [mixup_pytorch](https://github.com/leehomyc/mixup_pytorch): A PyTorch implementation of the paper Mixup: Beyond Empirical Risk Minimization in PyTorch.
175. <kbd>1000-</kbd> [inplace_abn](https://github.com/mapillary/inplace_abn): In-Place Activated BatchNorm for Memory-Optimized Training of DNNs
176. <kbd>1000-</kbd> [pytorch-pose-hg-3d](https://github.com/xingyizhou/pytorch-pose-hg-3d): PyTorch implementation for 3D human pose estimation
177. <kbd>1000-</kbd> [nmn-pytorch](https://github.com/HarshTrivedi/nmn-pytorch): Neural Module Network for VQA in Pytorch.
178. <kbd>1000-</kbd> [bytenet](https://github.com/kefirski/bytenet): Pytorch implementation of bytenet from "Neural Machine Translation in Linear Time" paper
179. <kbd>1000-</kbd> [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa): vqa, bottom-up-attention, pytorch
180. <kbd>1000-</kbd> [yolo2-pytorch](https://github.com/ruiminshen/yolo2-pytorch): The YOLOv2 is one of the most popular one-stage object detector. This project adopts PyTorch as the developing framework to increase productivity, and utilize ONNX to convert models into Caffe 2 to benifit engineering deployment.
181. <kbd>1000-</kbd> [reseg-pytorch](https://github.com/Wizaron/reseg-pytorch): PyTorch Implementation of ReSeg (arxiv.org/pdf/1511.07053.pdf)
182. <kbd>1000-</kbd> [binary-stochastic-neurons](https://github.com/Wizaron/binary-stochastic-neurons): Binary Stochastic Neurons in PyTorch.
183. <kbd>1000-</kbd> [pytorch-pose-estimation](https://github.com/DavexPro/pytorch-pose-estimation): PyTorch Implementation of Realtime Multi-Person Pose Estimation project.
184. <kbd>1000-</kbd> [interaction_network_pytorch](https://github.com/higgsfield/interaction_network_pytorch): Pytorch Implementation of Interaction Networks for Learning about Objects, Relations and Physics.
185. <kbd>1000-</kbd> [NoisyNaturalGradient](https://github.com/wlwkgus/NoisyNaturalGradient): Pytorch Implementation of paper "Noisy Natural Gradient as Variational Inference". 
186. <kbd>1000-</kbd> [ewc.pytorch](https://github.com/moskomule/ewc.pytorch): An implementation of Elastic Weight Consolidation (EWC), proposed in James Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks 2016(10.1073/pnas.1611835114).
187. <kbd>1000-</kbd> [pytorch-zssr](https://github.com/jacobgil/pytorch-zssr): PyTorch implementation of 1712.06087 "Zero-Shot" Super-Resolution using Deep Internal Learning
188. <kbd>1000-</kbd> [deep_image_prior](https://github.com/atiyo/deep_image_prior): An implementation of image reconstruction methods from Deep Image Prior (Ulyanov et al., 2017) in PyTorch.
189. <kbd>1000-</kbd> [pytorch-transformer](https://github.com/leviswind/pytorch-transformer): pytorch implementation of Attention is all you need.
190. <kbd>1000-</kbd> [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding): This is a PyTorch implementation of the AAAI-18 paper Gated-Attention Architectures for Task-Oriented Language Grounding
191. <kbd>1000-</kbd> [deep-forecast-pytorch](https://github.com/Wizaron/deep-forecast-pytorch): Wind Speed Prediction using LSTMs in PyTorch (arxiv.org/pdf/1707.08110.pdf)
192. <kbd>1000-</kbd> [cat-net](https://github.com/utiasSTARS/cat-net):  Canonical Appearance Transformations
193. <kbd>1000-</kbd> [minimal_glo](https://github.com/tneumann/minimal_glo): Minimal PyTorch implementation of Generative Latent Optimization from the paper "Optimizing the Latent Space of Generative Networks"
194. <kbd>1000-</kbd> [LearningToCompare-Pytorch](https://github.com/dragen1860/LearningToCompare-Pytorch): Pytorch Implementation for Paper: Learning to Compare: Relation Network for Few-Shot Learning. 
195. <kbd>1100+</kbd> [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings): PyTorch implementation of the NIPS-17 paper "Poincaré Embeddings for Learning Hierarchical Representations". 
196. <kbd>null</kbd> [pytorch-trpo(Hessian-vector product version)](https://github.com/ikostrikov/pytorch-trpo): This is a PyTorch implementation of "Trust Region Policy Optimization (TRPO)" with exact Hessian-vector product instead of finite differences approximation.
197. <kbd>1000-</kbd> [ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch): A PyTorch Implementation of Gated Graph Sequence Neural Networks (GGNN). 
198. <kbd>1000-</kbd> [visual-interaction-networks-pytorch](https://github.com/Mrgemy95/visual-interaction-networks-pytorch): This's an implementation of deepmind Visual Interaction Networks paper using pytorch
199. <kbd>1000-</kbd> [adversarial-patch](https://github.com/jhayes14/adversarial-patch): PyTorch implementation of adversarial patch. 
200. <kbd>1000-</kbd> [Prototypical-Networks-for-Few-shot-Learning-PyTorch](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch): Implementation of Prototypical Networks for Few Shot Learning (arxiv.org/abs/1703.05175) in Pytorch
201. <kbd>1000-</kbd> [Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch](https://github.com/orobix/Visual-Feature-Attribution-Using-Wasserstein-GANs-Pytorch): Implementation of Visual Feature Attribution using Wasserstein GANs (arxiv.org/abs/1711.08998) in PyTorch.
202. <kbd>1000-</kbd> [PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch](https://github.com/Blade6570/PhotographicImageSynthesiswithCascadedRefinementNetworks-Pytorch): Photographic Image Synthesis with Cascaded Refinement Networks - Pytorch Implementation
203. <kbd>1900+</kbd> [ENAS-pytorch](https://github.com/carpedm20/ENAS-pytorch): PyTorch implementation of "Efficient Neural Architecture Search via Parameters Sharing". 
204. <kbd>1000-</kbd> [Neural-IMage-Assessment](https://github.com/kentsyx/Neural-IMage-Assessment): A PyTorch Implementation of Neural IMage Assessment. 
205. <kbd>1000-</kbd> [proxprop](https://github.com/tfrerix/proxprop): Proximal Backpropagation - a neural network training algorithm that takes implicit instead of explicit gradient steps.
206. <kbd>9800+</kbd> [FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle): A Closed-form Solution to Photorealistic Image Stylization
207. <kbd>1000-</kbd> [Deep-Image-Analogy-PyTorch](https://github.com/Ben-Louis/Deep-Image-Analogy-PyTorch): A python implementation of Deep-Image-Analogy based on pytorch.
208. <kbd>1400+</kbd> [Person-reID_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch): PyTorch for Person re-ID. 
209. <kbd>1000-</kbd> [pt-dilate-rnn](https://github.com/zalandoresearch/pt-dilate-rnn): Dilated RNNs in pytorch. 
210. <kbd>1000-</kbd> [pytorch-i-revnet](https://github.com/jhjacobsen/pytorch-i-revnet): Pytorch implementation of i-RevNets.
211. <kbd>1000-</kbd> [OrthNet](https://github.com/Orcuslc/OrthNet): TensorFlow and PyTorch layers for generating Orthogonal Polynomials.
212. <kbd>1000-</kbd> [DRRN-pytorch](https://github.com/jt827859032/DRRN-pytorch): An implementation of Deep Recursive Residual Network for Super Resolution (DRRN), CVPR 2017
213. <kbd>1000-</kbd> [shampoo.pytorch](https://github.com/moskomule/shampoo.pytorch): An implementation of shampoo.
214. <kbd>1000-</kbd> [Neural-IMage-Assessment 2](https://github.com/truskovskiyk/nima.pytorch): A PyTorch Implementation of Neural IMage Assessment.
215. <kbd>1800+</kbd> [TCN](https://github.com/locuslab/TCN): Sequence modeling benchmarks and temporal convolutional networks locuslab/TCN
216. <kbd>1000-</kbd> [DCC](https://github.com/shahsohil/DCC): This repository contains the source code and data for reproducing results of Deep Continuous Clustering paper.
217. <kbd>1000-</kbd> [packnet](https://github.com/arunmallya/packnet): Code for PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning arxiv.org/abs/1711.05769
218. <kbd>1000-</kbd> [PyTorch-progressive_growing_of_gans](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans): PyTorch implementation of Progressive Growing of GANs for Improved Quality, Stability, and Variation.
219. <kbd>1000-</kbd> [nonauto-nmt](https://github.com/salesforce/nonauto-nmt): PyTorch Implementation of "Non-Autoregressive Neural Machine Translation"
220. <kbd>4000+</kbd> [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN): PyTorch implementations of Generative Adversarial Networks.
221. <kbd>1000-</kbd> [PyTorchWavelets](https://github.com/tomrunia/PyTorchWavelets): PyTorch implementation of the wavelet analysis found in Torrence and Compo (1998)
222. <kbd>1000-</kbd> [pytorch-made](https://github.com/karpathy/pytorch-made): MADE (Masked Autoencoder Density Estimation) implementation in PyTorch
223. <kbd>1000-</kbd> [VRNN](https://github.com/emited/VariationalRecurrentNeuralNetwork): Pytorch implementation of the Variational RNN (VRNN), from A Recurrent Latent Variable Model for Sequential Data.
224. <kbd>1000-</kbd> [flow](https://github.com/emited/flow): Pytorch implementation of ICLR 2018 paper Deep Learning for Physical Processes: Integrating Prior Scientific Knowledge.
225. <kbd>1000+</kbd> [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch): PyTorch implementation of convolutional networks-based text-to-speech synthesis models
226. <kbd>1000-</kbd> [psmm](https://github.com/elanmart/psmm): imlementation of the the Pointer Sentinel Mixture Model, as described in the paper by Stephen Merity et al.
227. <kbd>1000+</kbd> [tacotron2](https://github.com/NVIDIA/tacotron2): Tacotron 2 - PyTorch implementation with faster-than-realtime inference.
228. <kbd>1000-</kbd> [AccSGD](https://github.com/rahulkidambi/AccSGD): Implements pytorch code for the Accelerated SGD algorithm.
229. <kbd>1000-</kbd> [QANet-pytorch](https://github.com/hengruo/QANet-pytorch): an implementation of QANet with PyTorch (EM/F1 = 70.5/77.2 after 20 epoches for about 20 hours on one 1080Ti card.)
230. <kbd>1000-</kbd> [ConvE](https://github.com/TimDettmers/ConvE): Convolutional 2D Knowledge Graph Embeddings
231. <kbd>1000-</kbd> [Structured-Self-Attention](https://github.com/kaushalshetty/Structured-Self-Attention): Implementation for the paper A Structured Self-Attentive Sentence Embedding, which is published in ICLR 2017: arxiv.org/abs/1703.03130 .
232. <kbd>1000-</kbd> [graphsage-simple](https://github.com/williamleif/graphsage-simple): Simple reference implementation of GraphSAGE.
233. <kbd>2500+</kbd> [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch): A pytorch implementation of Detectron. Both training from scratch and inferring directly from pretrained Detectron weights are available.
234. <kbd>1000-</kbd> [R2Plus1D-PyTorch](https://github.com/irhumshafkat/R2Plus1D-PyTorch): PyTorch implementation of the R2Plus1D convolution based ResNet architecture described in the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
235. <kbd>1000-</kbd> [StackNN](https://github.com/viking-sudo-rm/StackNN): A PyTorch implementation of differentiable stacks for use in neural networks.
236. <kbd>1000-</kbd> [translagent](https://github.com/facebookresearch/translagent): Code for Emergent Translation in Multi-Agent Communication.
237. <kbd>1000-</kbd> [ban-vqa](https://github.com/jnhwkim/ban-vqa): Bilinear attention networks for visual question answering. 
238. <kbd>1000+</kbd> [pytorch-openai-transformer-lm](https://github.com/huggingface/pytorch-openai-transformer-lm): This is a PyTorch implementation of the TensorFlow code provided with OpenAI's paper "Improving Language Understanding by Generative Pre-Training" by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
239. <kbd>1000-</kbd> [T2F](https://github.com/akanimax/T2F): Text-to-Face generation using Deep Learning. This project combines two of the recent architectures StackGAN and ProGAN for synthesizing faces from textual descriptions.
240. <kbd>1000-</kbd> [pytorch - fid](https://github.com/mseitzer/pytorch-fid): A Port of Fréchet Inception Distance (FID score) to PyTorch
241. <kbd>1000-</kbd> [vae_vpflows](https://github.com/jmtomczak/vae_vpflows):Code in PyTorch for the convex combination linear IAF and the Householder Flow, J.M. Tomczak & M. Welling jmtomczak.github.io/deebmed.html
242. <kbd>1000-</kbd> [CoordConv-pytorch](https://github.com/mkocabas/CoordConv-pytorch): Pytorch implementation of CoordConv introduced in 'An intriguing failing of convolutional neural networks and the CoordConv solution' paper. (arxiv.org/pdf/1807.03247.pdf)
243. <kbd>1000-</kbd> [SDPoint](https://github.com/xternalz/SDPoint): Implementation of "Stochastic Downsampling for Cost-Adjustable Inference and Improved Regularization in Convolutional Networks", published in CVPR 2018. 
244. <kbd>1000-</kbd> [SRDenseNet-pytorch](https://github.com/wxywhu/SRDenseNet-pytorch): SRDenseNet-pytorch（ICCV_2017）
245. <kbd>1000-</kbd> [GAN_stability](https://github.com/LMescheder/GAN_stability): Code for paper "Which Training Methods for GANs do actually Converge? (ICML 2018)"
246. <kbd>1000-</kbd> [Mask-RCNN](https://github.com/wannabeOG/Mask-RCNN): A PyTorch implementation of the architecture of Mask RCNN, serves as an introduction to working with PyTorch
247. <kbd>1000-</kbd> [pytorch-coviar](https://github.com/chaoyuaw/pytorch-coviar): Compressed Video Action Recognition
248. <kbd>1000-</kbd> [PNASNet.pytorch](https://github.com/chenxi116/PNASNet.pytorch): PyTorch implementation of PNASNet-5 on ImageNet. 
249. <kbd>1000-</kbd> [NALU-pytorch](https://github.com/kevinzakka/NALU-pytorch): Basic pytorch implementation of NAC/NALU from Neural Arithmetic Logic Units arxiv.org/pdf/1808.00508.pdf
250. <kbd>1000-</kbd> [LOLA_DiCE](https://github.com/alexis-jacq/LOLA_DiCE): Pytorch implementation of LOLA (arxiv.org/abs/1709.04326) using DiCE (arxiv.org/abs/1802.05098)
251. <kbd>1000-</kbd> [generative-query-network-pytorch](https://github.com/wohlert/generative-query-network-pytorch): Generative Query Network (GQN) in PyTorch as described in "Neural Scene Representation and Rendering"
252. <kbd>1000-</kbd> [pytorch_hmax](https://github.com/wmvanvliet/pytorch_hmax): Implementation of the HMAX model of vision in PyTorch.
253. <kbd>1000-</kbd> [FCN-pytorch-easiest](https://github.com/yunlongdong/FCN-pytorch-easiest): trying to be the most easiest and just get-to-use pytorch implementation of FCN (Fully Convolotional Networks)
254. <kbd>1000-</kbd> [transducer](https://github.com/awni/transducer): A Fast Sequence Transducer Implementation with PyTorch Bindings.
255. <kbd>1000-</kbd> [AVO-pytorch](https://github.com/artix41/AVO-pytorch): Implementation of Adversarial Variational Optimization in PyTorch.
256. <kbd>1000-</kbd> [HCN-pytorch](https://github.com/huguyuehuhu/HCN-pytorch): A pytorch reimplementation of { Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation }.
257. <kbd>1000-</kbd> [binary-wide-resnet](https://github.com/szagoruyko/binary-wide-resnet): PyTorch implementation of Wide Residual Networks with 1-bit weights by McDonnel (ICLR 2018)
258. <kbd>1000-</kbd> [piggyback](https://github.com/arunmallya/piggyback): Code for Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights arxiv.org/abs/1801.06519
259. <kbd>6700+</kbd> [vid2vid](https://github.com/NVIDIA/vid2vid): Pytorch implementation of our method for high-resolution (e.g. 2048x1024) photorealistic video-to-video translation.
260. <kbd>1000-</kbd> [poisson-convolution-sum](https://github.com/cranmer/poisson-convolution-sum): Implements an infinite sum of poisson-weighted convolutions
261. <kbd>1000-</kbd> [tbd-nets](https://github.com/davidmascharka/tbd-nets): PyTorch implementation of "Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning" arxiv.org/abs/1803.05268 
262. <kbd>1000-</kbd> [attn2d](https://github.com/elbayadm/attn2d): Pervasive Attention: 2D Convolutional Networks for Sequence-to-Sequence Prediction
263. <kbd>2100+</kbd> [yolov3](https://github.com/ultralytics/yolov3): YOLOv3: Training and inference in PyTorch pjreddie.com/darknet/yolo
264. <kbd>1000-</kbd> [deep-dream-in-pytorch](https://github.com/duc0/deep-dream-in-pytorch): Pytorch implementation of the DeepDream computer vision algorithm. 
265. <kbd>1000-</kbd> [pytorch-flows](https://github.com/ikostrikov/pytorch-flows): PyTorch implementations of algorithms for density estimation
266. <kbd>1000-</kbd> [quantile-regression-dqn-pytorch](https://github.com/ars-ashuha/quantile-regression-dqn-pytorch): Quantile Regression DQN a Minimal Working Example
267. <kbd>1000-</kbd> [relational-rnn-pytorch](https://github.com/L0SG/relational-rnn-pytorch): An implementation of DeepMind's Relational Recurrent Neural Networks in PyTorch.
268. <kbd>1000-</kbd> [DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch): Deep Extreme Cut http://www.vision.ee.ethz.ch/~cvlsegmentation/dextr
269. <kbd>1000-</kbd> [PyTorch_GBW_LM](https://github.com/rdspring1/PyTorch_GBW_LM): PyTorch Language Model for Google Billion Word Dataset.
270. <kbd>1000-</kbd> [Pytorch-NCE](https://github.com/Stonesjtu/Pytorch-NCE): The Noise Contrastive Estimation for softmax output written in Pytorch
271. <kbd>1000-</kbd> [generative-models](https://github.com/shayneobrien/generative-models): Annotated, understandable, and visually interpretable PyTorch implementations of: VAE, BIRVAE, NSGAN, MMGAN, WGAN, WGANGP, LSGAN, DRAGAN, BEGAN, RaGAN, InfoGAN, fGAN, FisherGAN. 
272. <kbd>1000-</kbd> [convnet-aig](https://github.com/andreasveit/convnet-aig): PyTorch implementation for Convolutional Networks with Adaptive Inference Graphs.
273. <kbd>1000-</kbd> [integrated-gradient-pytorch](https://github.com/TianhongDai/integrated-gradient-pytorch): This is the pytorch implementation of the paper - Axiomatic Attribution for Deep Networks.
274. <kbd>1000-</kbd> [MalConv-Pytorch](https://github.com/Alexander-H-Liu/MalConv-Pytorch): Pytorch implementation of MalConv. 
275. <kbd>1000-</kbd> [trellisnet](https://github.com/locuslab/trellisnet): Trellis Networks for Sequence Modeling
276. <kbd>1000-</kbd> [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://github.com/minqi/learning-to-communicate-pytorch): pytorch implementation of  Learning to Communicate with Deep Multi-Agent Reinforcement Learning paper.
277. <kbd>1000-</kbd> [pnn.pytorch](https://github.com/michaelklachko/pnn.pytorch): PyTorch implementation of CVPR'18 - Perturbative Neural Networks http://xujuefei.com/pnn.html.
278. <kbd>1000-</kbd> [Face_Attention_Network](https://github.com/rainofmine/Face_Attention_Network): Pytorch implementation of face attention network as described in Face Attention Network: An Effective Face Detector for the Occluded Faces.
279. <kbd>1100+</kbd> [waveglow](https://github.com/NVIDIA/waveglow): A Flow-based Generative Network for Speech Synthesis.
280. <kbd>1000-</kbd> [deepfloat](https://github.com/facebookresearch/deepfloat): This repository contains the SystemVerilog RTL, C++, HLS (Intel FPGA OpenCL to wrap RTL code) and Python needed to reproduce the numerical results in "Rethinking floating point for deep learning" 
281. <kbd>1000-</kbd> [EPSR](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw): Pytorch implementation of [Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network](https://arxiv.org/pdf/1811.00344.pdf). This work has won the first place in PIRM2018-SR competition (region 1) held as part of the ECCV 2018.
282. <kbd>1000-</kbd> [ClariNet](https://github.com/ksw0306/ClariNet): A Pytorch Implementation of ClariNet arxiv.org/abs/1807.07281
283. <kbd>11700+</kbd> [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT): PyTorch version of Google AI's BERT model with script to load Google's pre-trained models
284. <kbd>1000-</kbd> [torch_waveglow](https://github.com/npuichigo/waveglow): A PyTorch implementation of the WaveGlow: A Flow-based Generative Network for Speech Synthesis. 
285. <kbd>2000+</kbd> [3DDFA](https://github.com/cleardusk/3DDFA): The pytorch improved re-implementation of TPAMI 2017 paper: Face Alignment in Full Pose Range: A 3D Total Solution.
286. <kbd>1000+</kbd> [loss-landscape](https://github.com/tomgoldstein/loss-landscape): loss-landscape Code for visualizing the loss landscape of neural nets.
287. <kbd>1000-</kbd> [famos](https://github.com/zalandoresearch/famos): 
Pytorch implementation of the paper "Copy the Old or Paint Anew? An Adversarial Framework for (non-) Parametric Image Stylization" available at http://arxiv.org/abs/1811.09236.
288. <kbd>1000-</kbd> [back2future.pytorch](https://github.com/anuragranj/back2future.pytorch): This is a Pytorch implementation of
Janai, J., Güney, F., Ranjan, A., Black, M. and Geiger, A., Unsupervised Learning of Multi-Frame Optical Flow with Occlusions. ECCV 2018.
289. <kbd>1000-</kbd> [FFTNet](https://github.com/mozilla/FFTNet): Unofficial Implementation of FFTNet vocode paper.
290. <kbd>1000-</kbd> [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch): A PyTorch Implementation of FaceBoxes.
291. <kbd>2000+</kbd> [Transformer-XL](https://github.com/kimiyoung/transformer-xl): Transformer-XL: Attentive Language Models Beyond a Fixed-Length Contexthttps://github.com/kimiyoung/transformer-xl
292. <kbd>1000-</kbd> [associative_compression_networks](https://github.com/jalexvig/associative_compression_networks): Associative Compression Networks for Representation Learning. 
293. <kbd>1000-</kbd> [fluidnet_cxx](https://github.com/jolibrain/fluidnet_cxx): FluidNet re-written with ATen tensor lib. 
294. <kbd>1000-</kbd> [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch): This repository contains PyTorch implementations of deep reinforcement learning algorithms.
295. <kbd>1000-</kbd> [Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch): This is a Pytorch implementation of faceplusplus's ShuffleNet-v2. 
296. <kbd>1000-</kbd> [GraphWaveletNeuralNetwork](https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork): This is a Pytorch implementation of Graph Wavelet Neural Network. ICLR 2019. 
297. <kbd>1000-</kbd> [AttentionWalk](https://github.com/benedekrozemberczki/AttentionWalk): This is a Pytorch implementation of Watch Your Step: Learning Node Embeddings via Graph Attention. NIPS 2018.
298. <kbd>1000-</kbd> [SGCN](https://github.com/benedekrozemberczki/SGCN): This is a Pytorch implementation of Signed Graph Convolutional Network. ICDM 2018.
299. <kbd>1000-</kbd> [SINE](https://github.com/benedekrozemberczki/SINE): This is a Pytorch implementation of SINE: Scalable Incomplete Network Embedding. ICDM 2018.
300. <kbd>1000-</kbd> [GAM](https://github.com/benedekrozemberczki/GAM): This is a Pytorch implementation of Graph Classification using Structural Attention. KDD 2018.
301. <kbd>1000-</kbd> [neural-style-pt](https://github.com/ProGamerGov/neural-style-pt): A PyTorch implementation of Justin Johnson's Neural-style.
302. <kbd>1000-</kbd> [TuckER](https://github.com/ibalazevic/TuckER): TuckER: Tensor Factorization for Knowledge Graph Completion.
303. <kbd>1000-</kbd> [pytorch-prunes](https://github.com/BayesWatch/pytorch-prunes): Pruning neural networks: is it time to nip it in the bud?
304. <kbd>1000-</kbd> [SimGNN](https://github.com/benedekrozemberczki/SimGNN): SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.
305. <kbd>1000-</kbd> [Character CNN](https://github.com/ahmedbesbes/character-based-cnn): PyTorch implementation of the Character-level Convolutional Networks for Text Classification paper. 
306. <kbd>1500+</kbd> [XLM](https://github.com/facebookresearch/XLM): PyTorch original implementation of Cross-lingual Language Model Pretraining.
307. <kbd>1000-</kbd> [DiffAI](https://github.com/eth-sri/diffai): A provable defense against adversarial examples and library for building compatible PyTorch models.
308. <kbd>1000-</kbd> [APPNP](https://github.com/benedekrozemberczki/APPNP): Combining Neural Networks with Personalized PageRank for Classification on Graphs. ICLR 2019.
309. <kbd>1000-</kbd> [NGCN](https://github.com/benedekrozemberczki/MixHop-and-N-GCN): A Higher-Order Graph Convolutional Layer. NeurIPS 2018.
310. <kbd>1000-</kbd> [gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch): Simple Text-Generator with OpenAI gpt-2 Pytorch Implementation
311. <kbd>1000-</kbd> [Splitter](https://github.com/benedekrozemberczki/Splitter): Splitter: Learning Node Representations that Capture Multiple Social Contexts. (WWW 2019).
312. <kbd>1000-</kbd> [CapsGNN](https://github.com/benedekrozemberczki/CapsGNN): Capsule Graph Neural Network. (ICLR 2019).
313. <kbd>1500+</kbd> [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch): The author's officially unofficial PyTorch BigGAN implementation.
314. <kbd>1000-</kbd> [ppo_pytorch_cpp](https://github.com/mhubii/ppo_pytorch_cpp): This is an implementation of the proximal policy optimization algorithm for the C++ API of Pytorch.
315. <kbd>1000-</kbd> [RandWireNN](https://github.com/seungwonpark/RandWireNN): Implementation of: "Exploring Randomly Wired Neural Networks for Image Recognition".
316. <kbd>1000-</kbd> [Zero-shot Intent CapsNet](https://github.com/joel-huang/zeroshot-capsnet-pytorch): GPU-accelerated PyTorch implementation of "Zero-shot User Intent Detection via Capsule Neural Networks".
317. <kbd>1000-</kbd> [SEAL-CI](https://github.com/benedekrozemberczki/SEAL-CI) Semi-Supervised Graph Classification: A Hierarchical Graph Perspective. (WWW 2019).
318. <kbd>1000-</kbd> [MixHop](https://github.com/benedekrozemberczki/MixHop-and-N-GCN): MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing. ICML 2019.
319. <kbd>1000-</kbd> [densebody_pytorch](https://github.com/Lotayou/densebody_pytorch): PyTorch implementation of CloudWalk's recent paper DenseBody.
320. <kbd>1000-</kbd> [voicefilter](https://github.com/mindslab-ai/voicefilter): Unofficial PyTorch implementation of Google AI's VoiceFilter system http://swpark.me/voicefilter. 
321. <kbd>1000-</kbd> [NVIDIA/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation): A PyTorch Implementation of [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://arxiv.org/abs/1812.01593), In CVPR2019. 
322. <kbd>1000-</kbd> [ClusterGCN](https://github.com/benedekrozemberczki/ClusterGCN): A PyTorch implementation of "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks" (KDD 2019).
323. <kbd>1000-</kbd> [NVlabs/DG-Net](https://github.com/NVlabs/DG-Net): A PyTorch implementation of "Joint Discriminative and Generative Learning for Person Re-identification" (CVPR19 Oral). 
324. <kbd>1000-</kbd> [NCRF](https://github.com/baidu-research/NCRF): Cancer metastasis detection with neural conditional random field (NCRF)
325. <kbd>1000-</kbd> [pytorch-sift](https://github.com/ducha-aiki/pytorch-sift): PyTorch implementation of SIFT descriptor. 
326. <kbd>1000-</kbd> [brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch): U-Net implementation in PyTorch for FLAIR abnormality segmentation in brain MRI. 
327. <kbd>1000-</kbd> [glow-pytorch](https://github.com/rosinality/glow-pytorch): PyTorch implementation of Glow, Generative Flow with Invertible 1x1 Convolutions (arxiv.org/abs/1807.03039) 
328. <kbd>1000-</kbd> [EfficientNets-PyTorch](https://github.com/zsef123/EfficientNets-PyTorch): A PyTorch implementation of EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
329. <kbd>1000-</kbd> [STEAL](https://github.com/nv-tlabs/STEAL): STEAL - Learning Semantic Boundaries from Noisy Annotations nv-tlabs.github.io/STEAL
330. <kbd>1000-</kbd> [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch): Official implementation of the ICML'19 paper "EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis".
331. <kbd>1000-</kbd> [Aspect-level-sentiment](https://github.com/ruidan/Aspect-level-sentiment): Code and dataset for ACL2018 paper "Exploiting Document Knowledge for Aspect-level Sentiment Classification"
332. <kbd>1000-</kbd> [breast_cancer_classifier](https://github.com/nyukat/breast_cancer_classifier): Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening arxiv.org/abs/1903.08297
333. <kbd>1000-</kbd> [DGC-Net](https://github.com/AaltoVision/DGC-Net): A PyTorch implementation of "DGC-Net: Dense Geometric Correspondence Network".

## Talks & conferences｜报告 & 会议

1. [PyTorch Conference 2018](https://developers.facebook.com/videos/2018/pytorch-developer-conference/): 2018年首届PyTorch开发者大会。

## Pytorch elsewhere ｜ Pytorch相关

1. <kbd>3900+</kbd> **[the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch)**: 不可思议的Pythorch：一份PyTorch相关的教程、论文、项目、社区等的清单。
2. <kbd>5400+</kbd> [generative models](https://github.com/wiseodd/generative-models): 各种生成模型，例如基于Pytorch和Tensorflow的GAN、VAE。 http://wiseodd.github.io  
3. [pytorch vs tensorflow](https://www.reddit.com/r/MachineLearning/comments/5w3q74/d_so_pytorch_vs_tensorflow_whats_the_verdict_on/): Reddit上的PyTorch和TensorFlow的比较文章。
4. [Pytorch discussion forum](https://discuss.pytorch.org/): PyTorch论坛。
5. <kbd>null</kbd> [pytorch notebook: docker-stack](https://hub.docker.com/r/escong/pytorch-notebook/): 类似于 [Jupyter Notebook Scientific Python Stack](https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook)
6. <kbd>1000-</kbd> [drawlikebobross](https://github.com/kendricktan/drawlikebobross): 使用神经网络作画！
7. <kbd>1000-</kbd> [pytorch-tvmisc](https://github.com/t-vi/pytorch-tvmisc): 该仓库收集了作者用PyTorch实现的各种玩意儿。
8. <kbd>1000-</kbd> [pytorch-a3c-mujoco](https://github.com/andrewliao11/pytorch-a3c-mujoco): 该项目旨在解决Mujoco中的控制问题，高度基于pytorch-a3c。
9. [PyTorch in 5 Minutes](https://www.youtube.com/watch?v=nbJ-2G2GXL0&list=WL&index=9).
10. <kbd>1000-</kbd> [pytorch_chatbot](https://github.com/jinfagang/pytorch_chatbot): 用PyTorch实现的聊天机器人。
11. <kbd>1000-</kbd> [malmo-challenge](https://github.com/Kaixhin/malmo-challenge): Malmo协作人工智能挑战-Pig Catcher团队。
12. <kbd>1000-</kbd> [sketchnet](https://github.com/jtoy/sketchnet): 指导计算机作画。http://www.jtoy.net/projects/sketchnet/
13. <kbd>1100+</kbd> [Deep-Learning-Boot-Camp](https://github.com/QuantScientist/Deep-Learning-Boot-Camp): 非盈利社区运营的5天深度学习训练营。 http://deep-ml.com.
14. <kbd>1000-</kbd> [Amazon_Forest_Computer_Vision](https://github.com/mratsim/Amazon_Forest_Computer_Vision): 亚马逊森林计算机视觉：使用PyTorch标记卫星图像标记/Keras中的PyTorch技巧。
15. <kbd>1600+</kbd> [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku): 用AlphaZero算法玩五子棋。
16. <kbd>null</kbd> [pytorch-cv](https://github.com/youansheng/pytorch-cv): null.
17. <kbd>1400+</kbd> [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid): Pytorch实现深度学习行人重新识别方法。
18. <kbd>1200+</kbd> [pytorch-template](https://github.com/victoresque/pytorch-template): PyTorch深度学习模版。
19. <kbd>1000-</kbd> [Deep Learning With Pytorch](https://github.com/svishnu88/DLwithPyTorch): 随书代码《[Deep Learning With Pytorch TextBook](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch)》 PyTorch实用指南：使用PyTorch建立文本和视觉神经网络模型。[亚马逊中国电子版](https://www.amazon.cn/dp/B078THDX3J/ref=sr_1_1?__mk_zh_CN=亚马逊网站&keywords=Deep+Learning+with+PyTorch&qid=1568007543&s=gateway&sr=8-1)
20. <kbd>1000-</kbd> [compare-tensorflow-pytorch](https://github.com/jalola/compare-tensorflow-pytorch): 比较用Tensorflow编写的层和用Pytorch编写的层之间的输出。
21. <kbd>1000-</kbd> [hasktorch](https://github.com/hasktorch/hasktorch): Haskell中的张量与神经网络。
22. [Deep Learning With Pytorch](https://www.manning.com/books/deep-learning-with-pytorch) Deep Learning with PyTorch 教你如何用Python和PyTorch实现深度学习算法。
23. <kbd>1000-</kbd> [nimtorch](https://github.com/fragcolor-xyz/nimtorch): PyTorch - Python + Nim，PyTorch的Nim前端。
24. <kbd>1000-</kbd> [derplearning](https://github.com/John-Ellis/derplearning): 自动驾驶遥控车代码。
25. <kbd>1000-</kbd> [pytorch-saltnet](https://github.com/tugstugi/pytorch-saltnet): Kaggle | TGS Salt Identification Challenge 第9名解决方案。
26. <kbd>1000-</kbd> [pytorch-scripts](https://github.com/peterjc123/pytorch-scripts): 一些脚本，使在Windows上使用PyTorch更加容易。
27. <kbd>1000-</kbd> [pytorch_misc](https://github.com/ptrblck/pytorch_misc): 为PyTorch讨论板创建的代码片段。
28. <kbd>1000-</kbd> [awesome-pytorch-scholarship](https://github.com/arnas/awesome-pytorch-scholarship): 收集了一系列优秀的PyTorch学术文章、指南、博客、课程和其他资源。
29. <kbd>1000-</kbd> [MentisOculi](https://github.com/mmirman/MentisOculi): PyTorch版raytracer。(raynet?)
30. <kbd>2400+</kbd> [DoodleMaster](https://github.com/karanchahal/DoodleMaster): “画出UI！”("Don't code your UI, Draw it !")
31. <kbd>1000-</kbd> [ocaml-torch](https://github.com/LaurentMazare/ocaml-torch): ocaml-torch为PyTorch张量库提供一些ocaml绑定。
32. <kbd>1000-</kbd> [extension-script](https://github.com/pytorch/extension-script): TorchScript自定义C++/CUDA运算符的示例。
33. <kbd>1000-</kbd> [pytorch-inference](https://github.com/zccyman/pytorch-inference):  Windows10 平台上 Pytorch 1.0在 C++ 中的推断。
34. <kbd>1000-</kbd> [pytorch-cpp-inference](https://github.com/Wizaron/pytorch-cpp-inference): 包含使用PyTorch C++ API执行推断的各种示例。
35. <kbd>1000-</kbd> [tch-rs](https://github.com/LaurentMazare/tch-rs): PyTorch的Rust绑定。
36. <kbd>1000-</kbd> [TorchSharp](https://github.com/interesaaat/TorchSharp): Pytorch引擎的.NET绑定。

**Feedback: If you have any ideas or you want any other content to be added to this list, feel free to contribute.**
