# deepext
Pytorch画像系の学習仕組み化ライブラリ

<br/><br/>

## 注意点
- データセットが指定したディレクトリになかった場合はダウンロードされます。
- データセットはtorchvision.datasetで使う形式です。
    - tarファイルと設定用フォルダなど含まれます
    - 手動でダウンロードした場合は形式が異なる場合があるので、スクリプトから自動ダウンロードをおすすめします
- コマンドの使い方は以下のコマンドで見れます
```
python train_---_demo.py -h
```
<br/>

## Setup
```
pip install -r requirements.txt
torch/torchvisionはこのサイトに従ってインストール(https://pytorch.org/)
```


## 学習
### 画像分類
```
python train_classification_demo.py  --model=efficientnet --dataset=stl --dataset_root=<STL10のパス(torchvision.dataset形式)> --progress_dir=<途中経過出力先ディレクトリ>
python train_classification_demo.py  --model=attention_branch_network --submodel=resnet18 --dataset=stl --dataset_root=<STL10のパス(torchvision.dataset形式)> --progress_dir=<途中経過出力先ディレクトリ>
```

<br/>

### 物体検出
```
python train_detection_demo.py --model=efficientdet --dataset=voc2012 --dataset_root=<Pascal VOCデータセット2012のパス> --progress_dir=<途中経過出力先ディレクトリ>
```

<br/>

### セグメンテーション
```
python train_segmentation_demo.py --moodel=custom_shelfnet --submodel=resnet18 --dataset=voc2012 --dataset_root=<Pascal VOCデータセット2007のパス(tarファイル)> --progress_dir=<途中経過出力先ディレクトリ> 
```

<br/>

### カメラを用いたリアルタイム推論
#### セグメンテーション
```
python camera_demo.py --model=custom_shelfnet --submodel=resnet18 --n_classes=21 --image_size=512 --label_names_path=voc_label_names.txt --load_weight_path=saved_weights\CustomShelfNet_epXXX.pth
```

#### 物体検出
```
python camera_demo.py --model=efficientdet --model_scale=0 --n_classes=20 --image_size=512 --label_names_path=voc_label_names.txt --load_weight_path=saved_weights\EfficientDetector_epXXX.pth
```

#### 分類
```
python camera_demo.py --model=mobilenet  --n_classes=10 --image_size=96 --label_names_path=stl_label_names.txt --load_weight_path=saved_weights\MobileNetV3_epXXX.pth
```

#### 分類(Attention map付き)
```
python camera_demo.py --model=attention_branch_network --submodel=resnet18 --n_classes=10 --image_size=96 --label_names_path=stl_label_names.txt --load_weight_path=saved_weights\AttentionBranchNetwork_epXXX.pth
```

<br/><br/>

## Using models
- ResNeSt
    - https://github.com/zhanghang1989/ResNeSt
- EfficientNet
    - https://github.com/lukemelas/EfficientNet-PyTorch
- EfficientDet
    - https://github.com/toandaominh1997/EfficientDet.Pytorch

## 学習経過
<img src="imgs/segmentation_progress.png" width="256" />
<img src="imgs/detection_progress.png" width="256" />