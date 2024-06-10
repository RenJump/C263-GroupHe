# C263-GroupHe: Efficient and Privacy-Preserving Synthetic Data Generation Using Consistency Models

## Project Overview
This project represents the culmination of efforts from UCLA's C263 course. We have innovated upon the latent diffusion model framework to design a fast tabular data generator that ensures privacy. Our approach introduces two major improvements:
1. Based on the property of decoupled training processes of the VAE and diffusion model components in latent diffusion model, we integrating DP-SGP within the VAE component to protect the privacy of latent variables.
2. We replaced the diffusion model component with a consistency model component to enable rapid generation of tabular data.

To our knowledge, this is the first attempt to apply a consistency model for tabular data generation.

## Data
We use data from the CTR Prediction - 2022 DIGIX Global AI Challenge as both our training and generation target. All related data, including the preprocessed raw data file `2022DIGIX.csv`, can be referred to at this [Google Drive link](https://drive.google.com/drive/folders/1_f1GnuCz-80aXst0R2soHnkI7WclhVYI?usp=sharing). Please download and preprocess the data as per the instructions provided in following sections to set up the project environment.

## Key Innovations
- **Differential Privacy**: Leveraging the opacus library, a PyTorch-based implementation of DP-SGD, we redesigned the encoder/decoder structure of the VAE component in the tabsyn model to accommodate privacy guarantees without storing individual gradients per batch.
- **Consistency Model**: Originally designed for image data, we adapted the consistency model for tabular data by developing a specialized data loader and substituting common U-Net components with MLPs suitable for tabular contexts.

## Code Structure
- `preprocess/`: Contains scripts for preprocessing the data from the 2022 DIGIX Global AI Challenge.
- `tabsyn/`: Includes code for training the VAE/Diffusion model with DP-SGD protection.
- `consistency_model.ipynb`: A notebook to train the consistency model, specifically configured to run in a Colab environment.
- `test/`: Contains notebooks to test the quality of synthesize data, including fidelity utility and privacy.

## How to Use

### Preparing the Data
1. **Download and Preprocess the Data:**
   - Download the training dataset from the 2022 DIGIX Global AI Challenge official website.
   - Execute the preprocessing scripts provided. Preprocessed files are already included in the folder for quick setup.

2. **Prepare Configuration Files:**
   - Create an `amazon.json` info file and place it under `tabsyn/data/Info`. This file is necessary for further data preprocessing tailored for the VAE/Diffusion model training.
   - Run the preprocessing script:
     ```
     python process_dataset.py
     ```

### Training & Synthesizing
3. **Train the VAE with DP-SGD Privacy Protection:**
   - Execute the following command to train the VAE model:
     ```
     python main.py --dataname amazon --method vae --mode train
     ```
   - Model parameters are saved under `tabsyn/tabsyn/vae/ckpt/amazon`. The `train_z.npy` file, which contains the latent vectors encoded from the training set, is crucial for subsequent model training.

4.1.1. **Train the Diffusion Model:**
   - After training the VAE, train the Diffusion model independently:
     ```
     python main.py --dataname amazon --method tabsyn --mode train
     ```

4.1.2. **Generate Tabular Data Using the Diffusion Model:**
   - Run the following command to generate data:
     ```
     python main.py --dataname amazon --method tabsyn --mode sample --sample_mode 0 --save_path [PATH_TO_SAVE]
     ```

4.2.1. **Train and Utilize the Consistency Model:**
   - After the VAE is trained, upload `train_z.npy` to Colab.
   - Follow the steps in `consistency_model.ipynb` to train the consistency model. Regenerate and download the `train_z.npy` file to replace the current version on your local machine. The file `sample_2step.npy` serves as an example, synthesized by a model trained for consistency using a two-step sampling process.

4.2.2. **Generate Tabular Data Using the Consistency Model:**
   - Generate data using the trained consistency model:
     ```
     python main.py --dataname amazon --method tabsyn --mode sample --sample_mode 1 --save_path [PATH_TO_SAVE]
     ```

All synthetic data are finally saved in `/tabsyn/synthetic/amazon`.
## Acknowledgments
Thanks to all contributors and the UCLA faculty who provided guidance and support throughout the course of this project.

## References
- [Link to tabsyn repository](https://github.com/amazon-science/tabsyn/tree/main)
- [Link to consistency models repository](https://github.com/Kinyugo/consistency_models)
- https://www.kaggle.com/datasets/xiaojiu1414/digix-global-ai-challenge

  
