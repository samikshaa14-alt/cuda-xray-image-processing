
# 🖥️ Image Colour Inversion Using CUDA C Programming

## 📌 Project Overview

This project demonstrates how CUDA C programming and GPU parallel processing can be applied to X-ray image visualisation through image colour inversion and pseudocolour mapping.

By transforming grayscale pixel intensities into inverted and colour-enhanced representations, the project explores ways to make subtle image intensity variations easier to observe.

The broader vision is to explore how GPU-accelerated image processing could contribute to enhanced medical image visualisation, potentially supporting the observation of subtle features, such as fine fracture lines, when developed and clinically validated further.

## 🎯 Objectives

- Implement image colour inversion using the formula `255 - pixel_value`.
- Apply pseudocolour mapping to grayscale X-ray images.
- Use CUDA threads to process independent pixels in parallel.
- Explore how alternative visual representations can highlight image intensity variations.
- Investigate the potential of GPU-accelerated processing for future medical image visualisation applications.

## 🛠️ Technologies Used

- CUDA C
- NVIDIA GPU Computing
- Parallel Programming
- Image Processing

## ⚙️ Implementation

1. Read the grayscale image as pixel intensity values.
2. Allocate GPU memory and transfer image data.
3. Launch CUDA kernels to process pixels in parallel.
4. Generate inverted and pseudocolour image outputs.
5. Compare the original image with the transformed results.

## 🖼️ Input and Output Results

- Check on the ppt for the input and output results!

## 💡 CUDA Concepts Applied

- CUDA kernel execution
- Parallel thread-based processing
- GPU memory allocation and data transfer
- CUDA event timing and synchronisation

## 🔬 Future Scope

- Explore additional image enhancement and contrast adjustment techniques.
- Investigate edge detection and feature enhancement methods.
- Evaluate processing performance and image quality on larger datasets.
- Explore the potential for highlighting subtle features in X-ray images through further development and testing.
- Investigate GPU-accelerated medical image visualisation with appropriate clinical validation.

## 🚀 Project Status

Completed CUDA C image-processing project exploring parallel pixel processing, image inversion, and pseudocolour mapping.

## ⚠️ Important Note

This project demonstrates image-processing and visualisation techniques for educational purposes. The current outputs have not been clinically validated for fracture detection or medical diagnosis. Any future clinical application would require rigorous technical evaluation and validation by qualified medical professionals.

