# COVID-19 Pneumonia Severity Classification

This repository contains the project on predicting the severity of COVID-19 pneumonia using chest X-ray (CXR) images. The project leverages convolutional neural networks (CNNs) to classify the severity based on the Brixia score, aiding in the early detection and management of the disease.

## Project Overview

The COVID-19 pandemic has overwhelmed healthcare systems globally. Chest X-rays provide a non-invasive method to monitor the progression of COVID-19 pneumonia. This project aims to develop an AI-assisted tool to predict the severity of pneumonia, thereby improving patient outcomes.

## Objectives

- Develop a machine learning model that predicts the severity of COVID-19 pneumonia from CXR images.
- Improve early detection and management of COVID-19 pneumonia to reduce mortality rates.
- Evaluate the model's accuracy and performance on a validation set.
- Deploy a CXR-based predictive model to optimize patient care and resource allocation.

## Methodology

The project is divided into several phases:

1. **Data Collection**: Utilize datasets such as the JSRT Database and small tuberculosis datasets from Shenzhen and Montgomery.
2. **Data Pre-processing**: Clean and prepare the dataset by removing noise, enhancing image quality, and resizing the images.
3. **Model Development**: Implement CNNs to learn features that distinguish different levels of pneumonia severity.
4. **Model Evaluation**: Test the model's generalization to unseen data and refine it as needed.
5. **Deployment**: Deploy the model for practical use in clinical settings.

## Libraries Used

- NumPy
- Matplotlib
- Seaborn
- TensorFlow

## Terminology

- **Brixia Score**: A semi-quantitative scoring system for assessing lung severity in COVID-19 patients based on CXR images.
- **Image Segmentation**: Process of dividing visual data into segments for specific processing.
- **Pneumonia Classification using CNN**: Utilizing CNNs to classify pneumonia severity from CXR images.

## Results

The model was evaluated on two datasets with different training epochs:

- **Dataset 1**: Utilized 10 epochs with early stopping.
- **Dataset 2**: Utilized 30 epochs.
- Confusion matrices were used to evaluate the model’s performance on both datasets.

## Conclusion

This project demonstrates the potential of AI in assisting healthcare professionals by accurately predicting COVID-19 pneumonia severity from CXR images. The developed model can be a valuable tool in improving patient care during the pandemic.

## References

1. [Tuberculosis Chest X-rays Montgomery Dataset](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery?resource=download)
2. [Tuberculosis Chest X-rays Shenzhen Dataset](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen)
3. [Nodules in Chest X-rays JSRT Dataset](https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt)
4. [NCBI Article on COVID-19](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7451075)
5. [COVID-19 Chest X-ray Paper on ArXiv](https://arxiv.org/abs/2005.11856)
6. [Pneumonia Detection Using CNNs](https://towardsdatascience.com/chest-x-rays-pneumonia-detection-using-convolutional-neural-network-63d6ec2d1dee)
