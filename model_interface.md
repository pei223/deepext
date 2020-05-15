# Basic interfaces of models.
## Classification
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, )

### Training output
- Tensor (batch size, class)

### Predict
- Numpy array (batch size, classes)


<br/><br/>


## Segmentation
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, class, height, width)

### Training output
- Tensor (batch size, class, height, width)

### Predict
- Numpy array(batch size, class, height, width)


<br/><br/>


## Object detection
### Training input
- image: Tensor (batch size, channel, height, width)
- annotation: Tensor (batch size, bounding box count, 5)
    - bounding box contains (x_min, y_min, x_max, y_max, class label)

### Training output
- Tuple (bounding box count, ), (bounding box count, ), (bounding box count, 4)
    - scores, classes, coordinates
    
### Predict
- Numpy array (batch size, class, bounding box count by class(variable length), 4)
    - bounding box contains (x_min, y_min, x_max, y_max)


<br/><br/>

