**Overview**
This project implements a time-series–based network congestion prediction framework for telecom traffic analysis. The system learns historical traffic behavior and predicts future network throughput to identify potential congestion scenarios before they occur.

**What is Network Congestion?**
Network congestion occurs when the volume of traffic exceeds the available network capacity, resulting in:
  - Increased latency
  - Packet loss
  - Reduced throughput
  - Performance degradation
Instead of detecting congestion after it occurs, this project focuses on predicting congestion proactively using traffic forecasting techniques.

**Objective**
  - The primary objective of this project is:
  - Predict future network throughput using historical traffic data
  - Detect potential congestion through predicted traffic behavior
  - Enable real-time congestion forecasting for multi-link telecom networks

**System Workflow**

Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering
   ↓
Sequence Generation (Sliding Window)
   ↓
Model Training
   ↓
Real-Time Data Pipeline
   ↓
Prediction / Inference
   ↓
Congestion Detection

**Input Features**

The model uses network traffic information such as:
  - Link_ID
  - Time
  - Throughput metrics
  - Traffic statistics

**Target variable:**
Moving Average Throughput

**Technologies Used**

**Programming**
Python
Machine Learning / Deep Learning
TensorFlow / Keras
Scikit-Learn
NumPy
Pandas
Data Pipeline & Streaming
Kafka
MongoDB

**Infrastructure**
  - HPC Cluster
  - Distributed Training
    
**Visualization**
  - Matplotlib

**Key Components**

**Data Processing** 
  - Missing value handling
  - Data cleaning
  - Feature scaling
  - Sequence generation

**Model Training**
  - Time-series forecasting model
  - Hyperparameter optimization
  - Distributed training support

**Real-Time Prediction Pipeline**
  - Streaming data ingestion
  - Database-backed inference
  - Continuous prediction generation
  - Congestion Prediction Logic

The system does not directly classify congestion.
Instead:
  - Historical traffic patterns are learned
  - Future throughput is predicted
  - Throughput degradation or abnormal predicted behavior is interpreted as congestion

**The framework demonstrates:**
  - Accurate throughput forecasting
  - Multi-link traffic prediction
  - Reduced training time using distributed learning
  - Real-time inference capability

**Future Improvements**
  - Larger-scale deployment
  - Enhanced real-time visualization
  - Advanced congestion thresholding
