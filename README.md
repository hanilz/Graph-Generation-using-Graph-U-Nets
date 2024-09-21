# Graph Generation Using Graph U-Nets

## By: [Hanil Zarbailov](https://www.linkedin.com/in/hanil-zarbailov-6598a4217/) & [Yana Komet](https://www.linkedin.com/in/yana-raitsin/) (LinkedIn)

## Mentor: Renata Avros

This repository serves as the official codebase for the final project named Graph Generation Using Graph U-Nets.

### Features

- Integrates Graph U-Nets for graph pooling instead of Misc-GAN's corasening module.
- Introduces two approaches for the reconstruction of the final synthetic graph:
  - G-Unpool on each synthetic graph simulated at each granularity layer and using the reconstruction module.
  - Using the reconstruction module on all synthetic graphs simulated at each granularity layer and then using G-Unpool on the final reconstructed synthetic graph. 
- Compatible with Google Colab for running experiments with external data.

### Installation

#### Prerequisites

To run this notebook, ensure you have the following installed:

- Python 3.10+
- Jupyter Notebook or Google Colab(we used Google Colab)
- Required Python libraries (listed below).
- Download a notebook from the **notebooks** folder.
- Download two datasets from the **datasets** folder:
  - One for using as the Graph U-Nets network at the **datasets/TrainedGraphUNets** folder
  - One for the simulation at the **datasets/Simulation** folder.

#### Required Libraries

Install the necessary Python libraries by running the following in your notebook or terminal:

`bash
pip install powerlaw
pip install python-igraph
pip install numpy==1.23.5  # Important for compatibility`

*Note: The version of NumPy is fixed to 1.23.5 due to compatibility issues with newer versions.*

### How to Use

#### Setup Google Drive Integration

If running in Google Colab, mount your Google Drive to access the necessary datasets or saved models.

`python
from google.colab import drive
drive.mount('/content/drive')`

Upload one of the notebooks from the folder named **notebooks** to Google Colab and the datasets you downloaded.

#### Running the Code

1. Open the notebook in Google Colab or Jupyter Notebook.

2. Upload the datasets you downloaded and update the paths.

3. Ensure all the necessary installations are complete.

4. Run each cell in sequence to mount your drive, load data, and execute the graph pooling methods integrated with Cycle-GAN.

### Args class
The Args class groups together essential hyperparameters for tuning the model. 

For example, the parameters control how many layers the model has, the size of feature vectors (latent dimensions), and how aggressively to regularize the model using dropout. The activation functions (for nodes and classification) are also set here, giving flexibility in how non-linearities are applied throughout the network. 

By encapsulating these parameters in an Args class, the model can be easily modified, tuned, and tested with different settings.

#### Attributes

- **`demo`**: 
  - A boolean flag to indicate if the model is running in demo mode.

- **`dataset_A`**: 
  - Path to the dataset file. If a specific version is provided, it dynamically appends the version number (e.g., `cora_1.mat`), otherwise defaults to `cora.mat`.

- **`checkpoint`**: 
  - Path where the model's training checkpoint will be saved. It includes the dataset version in the path if specified.

- **`filename`**: 
  - Name of the output network file. This also includes the dataset version if specified, otherwise defaults to `cora_output_network`.

- **`output_dir`**: 
  - Directory where the output results are saved. Includes the dataset version in the directory path if specified.

- **`epoch`**: 
  - Number of training epochs for the model. Controls how many times the model will see the entire dataset during training.

- **`layer`**: 
  - Defines the number of layers in the neural network.

- **`clusters`**: 
  - Number of clusters used during the clustering process. This attribute typically affects how the graph data is grouped.

- **`use_resnet`**: 
  - Boolean flag indicating whether to use residual blocks (ResNet architecture) in the generation network.

- **`use_lsgan`**: 
  - Boolean flag indicating whether to use Least Squares GAN (LSGAN) for the GAN loss function.

- **`which_direction`**: 
  - Defines the direction of the GAN transformation. It can be either 'AtoB' or 'BtoA' depending on whether data from domain A is mapped to domain B, or vice versa.

- **`which_stage`**: 
  - Specifies the current stage of the model, which could be 'training' or 'testing'.

- **`starting_layer`**: 
  - Specifies the layer from which the training/testing should start. Typically useful for fine-tuning or resuming from a certain depth in the network.

- **`shuffle`**: 
  - A boolean flag that indicates whether the graph data should be shuffled before training.

- **`gpu`**: 
  - Boolean flag indicating whether to use the GPU for training the model.

- **`kernel_number`**: 
  - Specifies the number of kernels (or filters) used in convolutional operations.

- **`iter`**: 
  - Number of iterations for the residual block function within the generator network. Controls how many times the generator operates on the graph during training.

### GNetArgs

The GNetArgs class is designed to store hyperparameters and configuration settings specific to the training of a Graph U-Nets network. 

This class allows for easy customization of key parameters such as learning rate, dropout rates, the number of layers, and activation functions, which are crucial for successfully training the model on graph data. 

By defining these attributes centrally, the class makes it easier to experiment with different settings and adjust them as needed.

#### Attributes

- **`seed`**: 
  - A random seed value used to ensure reproducibility of results. By setting a fixed seed, the randomness in the model training process is controlled, allowing for consistent results across different runs.

- **`data`**: 
  - The dataset name, in this case, `COLLAB`, which is a dataset commonly used for graph classification tasks.

- **`fold`**: 
  - Specifies the fold index in k-fold cross-validation. This is useful when evaluating the model's performance across different training and testing splits.

- **`num_epochs`**: 
  - Number of training epochs. This defines how many times the model will see the entire dataset during training. Increasing the number of epochs can allow the model to learn more but might lead to overfitting if set too high.

- **`batch`**: 
  - Batch size, which is the number of graphs processed together in one forward and backward pass. A smaller batch size is used here due to the potential size of the graph data.

- **`lr`**: 
  - Learning rate, which controls how much the model's weights are adjusted with respect to the loss gradient. A lower value generally leads to more stable convergence but slower training.

- **`deg_as_tag`**: 
  - A flag indicating whether to use node degrees as features or tags. This is particularly useful in graph neural networks to represent nodes using their degrees.

- **`l_num`**: 
  - Number of layers in the neural network. A greater number of layers allows the model to capture more complex patterns in the graph data.

- **`h_dim`**: 
  - Hidden dimension size, which defines the number of units in the hidden layers of the neural network. A larger hidden dimension allows for the learning of more complex representations but can increase computational requirements.

- **`l_dim`**: 
  - Latent dimension size, which is the dimensionality of the node embeddings after processing through the network.

- **`drop_n`**: 
  - Dropout rate for node features. Dropout is a regularization technique where a certain percentage of node features are randomly set to zero during training to prevent overfitting.

- **`drop_c`**: 
  - Dropout rate for convolutional layers. Dropout applied to convolutional layers ensures that the model does not become overly reliant on any particular set of graph features.

- **`act_n`**: 
  - Activation function used after processing node embeddings through each layer of the neural network. Common choices include `ReLU`, `Sigmoid`, or `Tanh`.

- **`act_c`**: 
  - Activation function applied after processing graph convolutional layers. It determines how the outputs of the convolutional layers are transformed.

- **`ks`**: 
  - A list of k-values used for graph pooling operations. These values typically represent the fraction of nodes retained in each graph pooling layer, which reduces the graph size while retaining key information.

- **`acc_file`**: 
  - The name of the file where accuracy results are saved. This file tracks model performance metrics during training and evaluation.

- **`save_model`**: 
  - Path where the trained model is saved after training. This allows the model to be reused or fine-tuned later without retraining from scratch.

- **`in_dim`**: 
  - Input dimension, which defines the number of input features for each node in the graph. For instance, if the input is node degree or some feature vector, this defines its dimensionality.


### Troubleshooting

#### Numpy Version Issues

If you encounter any issues related to numpy, ensure you are using version 1.23.5 as specified in the notebook.

### Reference GitHub Repositories

- [Graph U-Nets](https://github.com/HongyangGao/Graph-U-Nets) 

- [Misc-GAN](https://github.com/Leo02016/Miscgan)
