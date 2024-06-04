# Knowledge Distillation based Spatio-Temporal MLP Model for Real-Time Traffic Flow Prediction

## About

The PyTorch implementation of ST-MLP from the paper Knowledge Distillation based Spatio-Temporal MLP Model for Real-Time Traffic Flow Prediction.

## Network Structure

![method](https://github.com/zhangjunfeng1234/ST-MLP/blob/master/method.jpg)
An overview of the frameworks. A: TempEncoder/ST-Decoder module. The TempEncoder incorporates high-dimensional temporal information into the node features, whereas the ST-Decoder is responsible for the reverse. B: Spatiotemporal Mixer module, which is used to let nodes have spatiotemporal information.

In the code above, we've used D2STGNN(https://github.com/zezhishao/D2STGNN) as the teacher model. However, you're free to modify the teacher model as per your requirements. 

## Dataset

The dataset is sourced from STGCN and DCRNN.
https://github.com/VeritasYin/STGCN_IJCAI-18
https://github.com/liyaguang/DCRNN

The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).
PeMSD7 was collected from Caltrans Performance Measurement System (PeMS) in real-time by over 39, 000 sensor stations, deployed across the major metropolitan areas of California state highway system. The dataset is also aggregated into 5-minute interval from 30-second data samples. We randomly select a medium and a large scale among the District 7 of California containing 228 and 1, 026 stations, labeled as PeMSD7(M) and PeMSD7(L), respectively, as data sources. The time range of PeMSD7 dataset is in the weekdays of May and June of 2012.


## Model Details

You'll obtain the teacher model (xxx.model).

`python run_teacher.py` 

You'll obtain the student model (xxx.model).

`python run_student.py`


## Results
![method](https://github.com/zhangjunfeng1234/ST-MLP/blob/master/image.png)


