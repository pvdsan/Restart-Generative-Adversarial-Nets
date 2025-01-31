# Improving Generative Adversarial Networks Using Novel Discriminator Training Techniques

This repository contains the code and implementation details for the project titled **"Improving Generative Adversarial Networks Using Novel Discriminator Training Techniques."** The project explores innovative approaches to enhance the training of Generative Adversarial Networks (GANs) by addressing prevalent issues such as mode collapse, diminished gradient, and non-convergence.

## 📜 Project Overview

Generative Adversarial Networks (GANs) have emerged as a powerful tool in the field of deep learning, particularly for generating new data samples from a given distribution. Despite their success, GANs face significant challenges, including:

- **Mode Collapse:** The generator produces limited varieties of samples.
- **Diminished Gradient:** Difficulty in improving the generator due to poor feedback from the discriminator.
- **Non-Convergence:** Failure to reach an optimal state where the generator and discriminator perform at their best.

This project aims to address these challenges through two novel training techniques:

1. **New Discriminator Old Generator (NDOG) Approach:** Trains a pre-trained generator with new discriminators, emphasizing inner details and improving image quality.
2. **AutoRestart Discriminator Training:** Periodically resets the discriminator during training, allowing the generator to enhance its output with minimal discriminator interference.

## 🎯 Objectives

The main objectives of this project are:

- To improve GAN training by overcoming mode collapse and non-convergence issues.
- To explore novel training methodologies that enhance the Fréchet Inception Distance (FID) score, a metric used for evaluating the quality of generated images.
- To compare and contrast the traditional GAN training methods with the proposed methods using visual perception and statistical evaluation.

## 📁 Project Structure

The project report is structured as follows:

1. **Project Introduction:** Background on GANs, their applications, and the motivation behind the study.
2. **Literature Survey:** Overview of related works and foundational concepts such as Convolutional Neural Networks (CNNs), Fréchet Inception Distance (FID), and Inception Score.
3. **Base Model Implementation:** Description of the baseline model (DCGAN) and its performance evaluation.
4. **Experiment 1: Using NDOG Approach:** Details the implementation and results of the New Discriminator Old Generator technique.
5. **Experiment 2: AutoRestart Discriminator:** Explanation of the AutoRestart Discriminator Training method and its effectiveness.
6. **Conclusion:** Summary of results, limitations, and potential future work.

## 🧩 Methodologies

### 1. **New Discriminator Old Generator (NDOG) Approach**

This method aims to incrementally train the GAN by pairing a pre-trained generator with new discriminators. The key idea is to allow the generator to refine specific details of the generated images while the new discriminators focus on those refinements.

**Advantages:**
- Improves image quality progressively.
- Reduces mode collapse by refining inner details.
- Allows targeted training for specific attributes.

### 2. **AutoRestart Discriminator Training**

In this approach, the discriminator is periodically reinitialized during training. This helps prevent the discriminator from becoming too powerful, which would make it hard for the generator to learn effectively.

**Key Features:**
- Dynamically adjusts the learning rate of the discriminator.
- Enhances the generator's ability to improve over iterations.
- Reduces training time compared to the NDOG approach.

## 💻 Hardware and Software Requirements

### **Hardware:**

- **Laptop:** MSI GF65 95SD
- **GPU:** Nvidia GEFORCE GTX 1660 Ti
- **CPU:** Intel i7 9750 H
- **Memory:** 16GB RAM

### **Software and Packages:**

- **Python Environment:** Miniconda
- **IDE:** Jupyter Notebook, Spyder
- **Libraries:** Numpy, Matplotlib, PyTorch, OpenCV

## 📊 Results

The primary evaluation metric for this project is the **Fréchet Inception Distance (FID)** score. The results are as follows:

- **Baseline Model:** FID scores of 109.42, 155.5, and 124.16 for 200, 300, and 400 epochs, respectively.
- **NDOG Approach:** Significant improvements in FID scores, with the best result at 101.09 for the 300_200 model.
- **AutoRestart Discriminator:** The 300_50 model achieved the lowest FID score of 94.64, indicating superior image quality compared to the baseline and other models.

### **Comparative Analysis:**

The proposed training techniques show substantial improvements over traditional GAN training, particularly in reducing FID scores and enhancing image quality. However, limitations include the need for dynamic tuning of hyperparameters and the potential for premature convergence.

## 📈 Visual Results

Sample generated images and their progression through different training techniques are included in the report. Detailed graphs and tables are provided to compare the performance of each model visually and statistically.

## 📝 Future Work

- Expand the dataset to include more diverse images.
- Integrate advanced architectures like Inception v3 and ResNet into the model.
- Experiment with different values for the `Epoch_reset_AT` parameter in the AutoRestart approach.
- Explore stacking NDOG iterations for further image quality enhancements.

## 📚 References

1. Ian Goodfellow et al., "Generative Adversarial Nets," NIPS, 2014.
2. A. Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks," arXiv, 2015.
3. Martin Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," 2017.

(Refer to the report for the complete list of references.)

## 🛠️ Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/Improving-GANs-Training.git
