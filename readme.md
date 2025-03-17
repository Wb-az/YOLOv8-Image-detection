The objective of this piece of work is to detect disease in pear leaves using deep learning techniques. Recently [ultralytics](https://github.com/ultralytics/ultralytics) has released the new YOLOv8 model which demonstrates high accuracy and speed for image detection in computer vision. 

[OpenVino](https://docs.ultralytics.com/integrations/openvino/) models accelerate the inference processes without affecting the performance of the model. Therefore, we obtained the best model and converted it to the quantized OpenVino format for a speedy inference.

The dataset can be obtained from Zenodo "A Dataset for Diagnosis and Monitoring Plant Disease" - [DiaMOS Plant (Fenu G and Malloci FM, 2017)](https://zenodo.org/records/5557313).

The code presented in this work is based on Ultralytics and Openvino tutorials and cover the following:

- Extracting the data - Exploratory Analysis
- Build a YoloPipe line to preprocess, detect and post-process images
- Initial exploring performance with Yolov8m and Yolov8s
- Based on the result optimise the model using ray tune
- Build a YoloPipe line to preprocess, detect and post-process images with OpenVino models
- Convert the best model from the grid search to OpenVino FPS32 - Evaluate performance
- Convert the best model to a serialized-quantized model(int8) - Evaluate the performance
- Live inference on video


The results of the best model performance on the test set are shown in the table below:


---


|Model | Class | Instances |  Box(P) | Box (R) | Box mAP50 | Box mAP50-95 |
|------|-------|--------|---------|---------|-----------|--------------|
| Best model|all   | 180    | 0.819  | 0.864   | 0.876     | 0.701        |
| OpenVino FP32|all   | 180    | 0.77  | 0.943   | 0.874     | 0.717        |
| Quantized OpenVino|all   |   180   | 0.774  | 0.95   | 0.876    | 0.715       |


The table above shows that the FPS32 and quantized OpenVino model did not affect the performance of the model at detecting disease in pear leaves but detected the disease faster than the no-serialised best model.



<center>
<table>
    <tr>
      <td>
      <img src="yolov8-pear-experiments/grid-opt/test/val_batch2_pred.jpg" width=500, height=600 >
      </td>
      <td>
      <img src="yolov8-pear-experiments/grid-opt/test/PR_curve.png" width=650, height=350>
      </td>
     </tr>
</table>
</center>



