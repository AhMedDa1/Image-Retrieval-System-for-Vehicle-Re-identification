# Image-Retrieval-System-for-Vehicle-Re-identification

This repository contains a Jupyter notebook for implementing an image retrieval system for vehicle re-identification using the VeRi dataset. The model leverages a pre-trained convolutional neural network (VGG16) to extract features and employs triplet loss for learning an embedding space to distinguish between vehicles effectively.

**Features**

**Pre-trained Model:** Uses VGG16 without the top fully connected layers to build a feature extraction pipeline.

**Triplet Loss:** Implements triplet loss with random triplet mining to train the embedding network.

**Evaluation:** Calculates mean average precision (mAP) and visualizes top-5 retrieval results for queries.


**Dataset**

VeRi Dataset: Images of vehicles for re-identification tasks.
Ensure the dataset is downloaded and paths to image_train, image_query, image_test, and XML label files are updated in the notebook.

**Overview of Approach**

1- Feature Extraction: Modified VGG16 for creating 128-dimensional embeddings.

2- Data Preparation:
   - Parsed XML files to group images by vehicle IDs.
   - Created random triplets (anchor, positive, negative) for training.
     
3- Training:
   - Optimized triplet loss with Adam optimizer.
   - Tuned hyperparameters, including margin and learning rate.

     
4- Evaluation:
   - Generated embeddings for query and test images.
   - Computed Euclidean distances to rank results.
   - Visualized retrieval results and calculated mAP.

     
**Requirements**
TensorFlow, NumPy, OpenCV, scikit-learn, Matplotlib, and LXML.


**Usage**
- Clone the repository.
- Install the required libraries.
- Run the notebook and follow the instructions provided in the markdown cells.
