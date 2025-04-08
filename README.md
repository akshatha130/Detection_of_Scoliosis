# Detection of Spinal Scoliosis
Scoliosis is a medical condition characterized by abnormal lateral curvature of the spine, affecting millions of people throughout the world. Early detection of scoliosis is important for good treatment and management. Conventional detection approaches are mostly based on manual observation of X-ray images, which in itself is laborious and prone to human error. The present project is concerned with the development of an AI system for automatic detection of scoliosis based on deep learning algorithms. 
A Convolutional Neural Network (CNN) classifies spine X-ray images into three different categories: Normal, Scoliosis, and Spondylosis. The model learns from a database of spine X-ray images with a test set accuracy of 81%. The system also utilizes Grad-CAM (gradient-weighted Class Activation Mapping) to provide visual explanations of the model's predictions and align them with human-level interpretation.
The project is a demonstration of AI working in medical imaging and aims to provide an early detection of scoliosis in a scalable manner. In the future, the aim is to enlarge the dataset, increase the accuracy of the model, and incorporate the system into clinical workflows.

<img src="https://github.com/user-attachments/assets/ce7006e8-630e-46ae-9c3c-8ee379a98330" height="300" width="400"/>
<img src="https://github.com/user-attachments/assets/d262f5c4-daa9-471f-acfa-1a6ac75b90e4" height="302"/>
    
## 1. The below image Shows the Comparison Between Training and Validation Accuracy:
<img src="https://github.com/user-attachments/assets/4bb034ea-34d9-4ba5-99f9-b1f5edd17395" height="310" width="515"/>

## 2. Grade-Cam-Visulation:
<img src="https://github.com/user-attachments/assets/f588f5ef-7752-4577-b85e-72b9e89b9368" height="360" width="515"/>


---

## 🚀 How to run it on your own machine

### 1. Install the requirements

```bash
pip install -r requirements.txt
```
### 2. Before running the app, you need to train the model

Run the project.ipynb file 

### 3. Run the Streamlit app <img src = "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" margin-top="20" height="50" width="100">
```bash
streamlit run app.py
```


