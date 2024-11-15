# COVID lungs detection using Decision tree and Logistic Regression

## Overview

  <table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e222132d-c328-41c6-805c-b5c5814e05c0" width="300"/><br/>
      <b>Normal lung Xray</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8f939b04-25ab-4eb5-bf1e-78e5de27505b" width="300"/><br/>
      <b>COVID lung Xray</b>
    </td>
  </tr>
</table>

## Data pre-processing 
Two ImageDataGenerator objects are instantiated to facilitate the preprocessing of image data for a neural network. These generators are part of the Keras library, which provides tools for data augmentation and preprocessing that are essential for image-based machine learning tasks.
Normalization is applied to both the training and testing image sets by rescaling pixel values with a coefficient of 1/255.  
The images undergo several preprocessing steps, which include:  
•	Resizing to a uniform dimension of 28×28 pixels.  
•	Conversion to grayscale format.  
•	Grouping into batches containing 200 images each.  
The datasets are prepared with a binary class mode, which is indicative of a two-class classification problem.  

## Logistic regression
**Model**
![image](https://github.com/user-attachments/assets/18efc46a-ae4f-4453-8c9c-8bf108aff536)
In the experimental setup, the batch size was uniformly set to 200 for both the training and testing phases. The training process was conducted over 10 epochs at the learning rate of 0.01. Comparative analysis of optimization algorithms indicated that the Adam optimization method outperformed traditional gradient descent when coupled with Principal Component Analysis (PCA). Furthermore, within the scope of the Adam optimizer framework, Ridge regression was employed due to its superior performance in comparison to the Lasso technique, which yielded suboptimal results.  
Principal Component Analysis (PCA) was employed in the gradient descent model to address the challenges posed by the dataset's high dimensionality, which includes 784 features. This approach was based on the rationale that a subset of these features might be redundant or highly collinear. In a parallel strategy for the Adam-optimized model, L2 regularization was applied with the intent to diminish the feature space's complexity. This regularization technique penalizes the square of the feature coefficients, effectively shrinking them and thus reducing the model's complexity. Both methods aim to enhance model performance by simplifying the feature space and mitigating the risk of overfitting.

**Performance**
  <table>
  <tr>
    <td align="center">
      <img src="![image](https://github.com/user-attachments/assets/0e472a81-6bf4-4a9e-946e-5716b770ed62)" width="300"/><br/>
      <b>ROC curve of Gradient descent model with PCA</b>
    </td>
    <td align="center">
      <img src="![image](https://github.com/user-attachments/assets/bb2428f8-61c8-431a-a7b6-b37d0dc4097f)" width="300"/><br/>
      <b>ROC curve of Adam model with L2 regression</b>
    </td>
  </tr>
</table>

The model employing gradient descent in conjunction with Principal Component Analysis (PCA) demonstrated superior performance, as evidenced by an Area Under the Curve (AUC) metric of 0.986. This outperformed the model utilizing the Adam optimization algorithm with L2 regularization (commonly referred to as Ridge regression), which achieved an AUC of 0.971. The higher AUC value associated with the gradient descent model suggests a more accurate classification capability when compared to the Adam model with L2 regression.

## Decision tree
**Model**
![image](https://github.com/user-attachments/assets/889b7641-f1bb-41d3-bab3-7b55d4cffa3c)
In contrast to the batch training methodology typically employed in logistic regression, the implementation of Principal Component Analysis (PCA) with the top 30 principal components resulted in a diminished accuracy of the model. This reduction in performance may be attributed to the exclusion of critical components that are integral to the dataset's inherent variance and predictive capacity. Consequently, to mitigate the potential loss of important information, the entire datasets for both training and testing were utilized in the construction and assessment of a decision tree model. The decision tree was configured with a maximum depth of 30, allowing for a more exhaustive representation of the data's feature space.

**Performance**
![image](https://github.com/user-attachments/assets/30d6ca43-4074-43c8-991f-4922d6c9b69a)

TP = 104, FP = 11, TN = 306, FN = 12  
Accuracy = (TP + TN) / (TP + TN  + FP + FN) = (104 + 306)/(104 + 306 + 11 + 12) = 0.947  
Precision = TP / (TP + FP) = 104 / (104 + 11) = 0.904  
Recall = TP / (TP + FN) = 104 / (104 + 12) = 0.897  

Despite the reduced training duration, the decision tree model exhibited inferior performance relative to the logistic regression model. Nonetheless, the architecture of the decision tree model was characterized by its simplicity, as it was constructed without the complexity of batch training procedures.


## Conclusion
**Discussion**
The fundamental distinction in the development processes of logistic regression and decision tree models lies in their training methodologies. Logistic regression typically employs an iterative approach, utilizing epochs and batch iterations to optimize the model's parameters. In contrast, the decision tree model eschews this iterative process, instead ingesting the entire dataset in a single pass for both training and testing. The complexity and capacity of the decision tree are regulated by adjusting its depth, which directly influences how the model learns and generalizes from the data. This difference in approach reflects the distinct underlying mechanisms of these two types of models.  
The decision tree model, while straightforward in its implementation, did not match the performance of the logistic regression model. Enhancing the resolution of the input images might improve the decision tree's efficacy; however, this would necessitate increased computational resources.  
Conversely, logistic regression demonstrated robust performance, evidenced by a high Receiver Operating Characteristic (ROC) curve. Both optimization methods, gradient descent, and Adam, yielded effective models, although the gradient descent approach necessitated a preliminary application of Principal Component Analysis (PCA) to manage the high-dimensional feature space efficiently.

**Future work**
For future endeavors aimed at augmenting model performance, a variety of image preprocessing techniques, including different resolutions and aspect ratios, should be explored. Additionally, fine-tuning other hyperparameters, such as the learning rate and the Adam optimizer's beta1 and beta2 parameters, could provide a deeper insight into the dataset's characteristics and potentially unveil opportunities for further enhancements. An evaluation of the training velocity is also recommended to reduce the duration of model training. This may involve a more granular analysis of feature selection and optimization of the batch size to achieve a more efficient training process.
