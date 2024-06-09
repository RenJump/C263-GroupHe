# C263-GroupHe: Fast Tabular Data Generator with Privacy Guarantee

## Project Overview
This project represents the culmination of efforts from UCLA's C263 course. We have innovated upon the latent diffusion model framework to design a fast tabular data generator that ensures privacy. Our approach introduces two major improvements:
1. We decoupled the training processes of the VAE and diffusion model components, integrating DP-SGP within the VAE component to protect the privacy of latent variables.
2. We replaced the diffusion model component with a consistency model component to enable rapid generation of tabular data.

To our knowledge, this is the first attempt to apply a consistency model for tabular data generation.

## Data
We use data from the CTR Prediction - 2022 DIGIX Global AI Challenge as both our training and generation target.

## Key Innovations
- **Differential Privacy**: Leveraging the opacus library, a PyTorch-based implementation of DP-SGD, we redesigned the encoder/decoder structure of the VAE component in the tabsyn model to accommodate privacy guarantees without storing individual gradients per batch.
- **Consistency Model**: Originally designed for image data, we adapted the consistency model for tabular data by developing a specialized data loader and substituting common U-Net components with MLPs suitable for tabular contexts.

## Code Structure
- `preprocess/`: Contains scripts for preprocessing the data from the 2022 DIGIX Global AI Challenge.
- `tabsyn/`: Includes code for training the VAE/Diffusion model with DP-SGD protection.
- `consistency_model.ipynb`: A notebook to train the consistency model, specifically configured to run in a Colab environment.

## Installation
Describe how to install and run the project here, including environment setup if specific versions of libraries or tools are necessary.

## How to Use
Provide examples on how the models can be trained and how the generated data can be used, possibly with code snippets or command-line examples.

## Contributions
Details on how others can contribute to the project, including how to submit issues, feature requests, and pull requests.

## License
Specify the license under which the project is made available.

## Acknowledgments
Thanks to all contributors and the UCLA faculty who provided guidance and support throughout the course of this project.

## References
- [Link to tabsyn repository](https://github.com/amazon-science/tabsyn/tree/main)
- [Link to consistency models repository](https://github.com/Kinyugo/consistency_models)
