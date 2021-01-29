DETECTION_DATASET_INFO = {
    "voc2012": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "voc2007": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
}

CLASSIFICATION_DATASET_INFO = {
    "stl10": {
        "label_names": ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'],
        "n_classes": 10,
        "size": (96, 96),
    },
    "cifar10": {
        "label_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "n_classes": 10,
        "size": (32, 32),
    }
}

SEGMENTATION_DATASET_INFO = {
    "voc2012": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "voc2007": {
        "label_names": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        "n_classes": 20,
    },
    "cityscape": {
        "label_names": ['ego vehicle', 'rectification', 'out of roi', 'static', 'dynamic',
                        'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building',
                        'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole',
                        'polegroup', 'traffic light', 'traffic sign', 'vegetation',
                        'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                        'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',
                        'license plate'],
        "n_classes": 34,
    }
}
