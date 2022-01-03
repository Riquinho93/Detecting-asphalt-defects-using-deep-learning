# Detecting asphalt defects using deep learning

## Abstract

Repairing defects present in road asphalt is an important task to increase the comfort and safety of road users. In addition, it allows to avoid intense levels of wear on the asphalt, which also lead to higher repair costs. In many cases, those responsible for paved road maintenance notice or observe road defects only when a serious accident occurs. And this could be avoided if there was greater concern with detecting and quickly repairing these irregularities on the highways. A more economical and profitable way is to prevent these road failures from becoming an obstacle. For this, it is necessary to identify the defects even before an accident occurs to defray the cost of maintenance and insecurity problems on the highways. Automating this process with object recognition techniques can bring greater safety in traffic, due to a better organization of roads that need maintenance. The objective of this work is to build a defect recognition algorithm in digital images and videos of asphalts through Deep learning. We will use Yolo as our object detector, as it can predict objects in a few seconds. Our dataset consists of 613 images, spread over 317 crack images and 296 hole images. Although defect recognition is considered a difficult task, we obtained an average accuracy of 96% in our best model.


Keywords: Yolo, object detection, Deep learning.


## Contents

- [Chapter 1 - Introduction](Chapter1-Introduction)
- 1.1 Framework 
- 1.2 Objectives 
- 1.3 Structure of the document 
Chapter 2 
- 2.1 Artificial Intelligence 
- 2.1.1 How does AI learn? 
- 2.2 Computer vision 
- 2.3 Machine Learning 
- 2.3.1 Tasks performed by machine learning 
- 2.4 Artificial neural networks 
- 2.5 Convolutional neural networks 
- 2.5.1 Convolution Operation 
- 2.5.2 Activation function 
- 2.5.3 Hyper parameters 
- 2.5.4 Transfer Learning 
- 2.6 Object Detection 
- 2.6.1 Yolo 
- 2.6.2 Yolov4 
- 2.7 Metrics 
- 2.7.1 Parameters in Object Detection 
- 2.7.2 Intersection over union (IoU) 
- 2.7.3 Precision 
- 2.7.4 Recall 
- 2.7.5 Average Accuracy (AP) 
- 2.7.6 mAP (mean average precision) 
- 2.7.7 F1-Score 
Chapter 3 
- 3.1 Project development stages 
- 3.2 Database 
- 3.3 Tools used 
- 3.4 Deployment in a Cloud Environment 
Chapter 4 
- 4.1 Pre-trained model 
- 4.2 Configuration of files and hyper parameters 
- 4.2.1 Configure the .cfg file 
- 4.2.2 Configure obj.names 
- 4.2.3 Configure train.txt and test.txt 
- 4.3 Training 
Chapter 5 
- 5.1 Dataset Analyzes 
- 5.2 Results between versions of yolo 
- 5.3 Testing with everyday data 
- 5.4 Test with possible obstacles for model 
- 5.5 Analysis of tests with videos 
- 5.6 Observations and analysis of results 
Chapter 6 
- 6.1 Recap and final remarks of the project 
- 6.2 Future work 


Chapter 1 - Introduction

Chapter 1 is dedicated to an introduction to the theme of the work, describing the general ideas of the problem and its importance. In addition, the objectives of the work and a structure of the report are explicit.

1.1 Framework

According to the World Bank, referenced in the G1 [1], among the main world economies, Brazil has the highest concentration of road transport for passengers and passengers. Not Brazil, 58% of cargo transport is carried out by road - then we have Australia with 53%, China with 50%, Russia with 43% and 8% in Canada. Finally, a road network is used to transport 75% of production in Brazil. These data report the great importance of highways for countries.

A survey carried out on Brazilian highways by the National Transport Confederation (CNT) reported in the newspaper Metrópoles [2], pointed out that 59% of the highways had some type of problem. The entity analyzed 108 miles of paved roads. Knowing this, the conservation of highways

The correction of the highways is necessary to guarantee that the cargo flow is carried out quickly and efficiently, accelerating the economic development of Brazil, as shown by Rodrigues and Colmenero [3].

In addition to the economic factor, the bad categories of the highways are considered a determining factor in the occurrence of traffic accidents [4]. And with irregular roads, they provide various damages to vehicles that can lead to various types of accidents.

It is important to consider that there are other factors that determine a safe road such as traffic signs. However, the focus of the work is on the defects of paved roads.

To automate this process of detecting defects in digital images on asphalt, we will use Deep Learning (DL) techniques [29]. DL or deep learning in Portuguese, is a sub-area of Machine Learning (ML), machine learning translating into Portuguese. DL is a successive representation of convolutional layers (topic explained in chapter 2). And the total number of layers, two or more, that are part of the data model is required of model depth. In DL, there are several parameters and concepts that make this work possible.

Deep learning is no longer a trend, it has become a reality. With DL, incredible projects are being created between different professional areas. For example, performing analysis of symptoms and causes and suggesting medications for patients; To train robotic mechanisms for people who do not have legs and arms, or to classify by means of images whether or not a patient has a specific disease. All these ideas are derived from the medical field.

In agriculture, DL can analyze and monitor vegetation 24x7 (24 hours for 7 days) to detect any disease and suggest a possible pesticide. You can also analyze how to change the weather and suggest a particular vegetable for that type of field and climate. Furthermore, it is able to classify fruits and vegetables after cultivation in terms of quality.

DL can act as a voice assistant, understand a sentence of a sentence to be able to answer back. DL acts in autonomous cars, with a vehicle, sensors and a global location of the location and determining that the vehicle must travel a path to the target. DL also acts as a research and marketing manager, managing to analyze each user and identify their interests, making advertisements targeted at target users.

These are some of the other applications that make DL fascinating. Throughout the document, what is and how deep learning works is detailed, in addition to explaining in detail the proposal chosen in this work.


1.2 Objective

In general, the objective of this work is to explore convolutional neural networks capable of detecting defects in asphalts through digital images and videos. Specifically, this work aims to:

- Conduct a study of the context and technologies for detecting defects on paved roads.
- Raise data sets for defects on paved roads, such as potholes and cracks.
- Implement and train with different algorithms for detecting objects in the collected data sets.
- Evaluate and compare the results for the continuation of future work.

1.3 Document structure

This document has a total of six chapters and is organized as follows:
In the first chapter, the introductory phase of the project is explained by contextualizing the framework, objective and structure of the report.
The second chapter is intended to produce scientific knowledge about the proposed solution to solve the problem. Among them are approaches and concepts on Artificial Intelligence, Computer Vision, Machine Learning, Deep Learning, architecture of used models and metrics.
The third chapter is responsible for detailing the tools used in the project, in addition to the step-by-step diagram to successfully complete the work and information about the database.
The fourth chapter is the development of the work, that is, the project implementation is described in sequence. This includes the files and settings that were made for a job run.
The fifth chapter represents an evaluation phase of the algorithm using metrics such as mAP and IoU. Also, there are several comparisons between the models chosen for the resolution of the project proposal.
The sixth chapter is the conclusion of all the work, prompting the undoing and the next step of the project.


