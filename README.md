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
python train_---.py -h
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
python train_classification.py --progress_dir="途中経過出力先ディレクトリ" --dataset_root="STL10のパス(torchvision.dataset形式) --model=efficientnet --dataset=stl"
```

<br/>

### 物体検出
```
python train_detection.py --progress_dir="途中経過出力先ディレクトリ" --dataset_root="Pascal VOCデータセット2012のパス" --dataset=voc2012 --model=efficientdet
```

<br/>

### セグメンテーション
```
python train_segmentation.py --progress_dir="途中経過出力先ディレクトリ" --dataset_root="Pascal VOCデータセット2007のパス(tarファイル) --moodel=pspnet --dataset=voc2007"
```

<br/>

### カメラを用いたリアルタイム推論
#### セグメンテーション
```
python camera_demo.py --model=custom_shelfnet  --n_classes=21 --load_weight_path=saved_weights\CustomShelfNet_epXXX.pth --image_size=512 --label_names_path=voc_label_names.txt
```

#### 物体検出
```
python camera_demo.py --model=efficientdet  --n_classes=20 --model_scale=0 --load_weight_path=saved_weights\EfficientDetector_epXXX.pth --image_size=512 --label_names_path=voc_label_names.txt
```

#### 分類
```
python camera_demo.py --model=mobilenet  --n_classes=10 --load_weight_path=saved_weights\MobileNetV3_epXXX.pth --image_size=96 --label_names_path=stl_label_names.txt
```

#### 分類(Attention map付き)
```
python camera_demo.py --model=attention_branch_network  --n_classes=10 --load_weight_path=saved_weights\AttentionBranchNetwork_epXXX.pth --image_size=96 --label_names_path=stl_label_names.txt
```

<br/><br/>


## 学習経過
<img src="imgs/segmentation_progress.png" width="256" />
<img src="imgs/detection_progress.png" width="256" />
