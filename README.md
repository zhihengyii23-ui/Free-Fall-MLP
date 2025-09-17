# MLP Free Fall Model
This repository contains a simple **Multi-Layer Perceptron (MLP)** model designed to predict free-fall time of an object, given its initial height and the gravitational acceleration of the planet (Earth, Mars, Jupiter).

It is intended as a **learning project** that combines physics with machine learning.

## Background

In Physics, the free fall kinematics equation is:
\[
t = \sqrt{\frac{2h}{g}}
\]

where 
- *t* = free fall time 
- *h* = initial height (meters)
- *g* = gravitational acceleration (m/s²)

We generate synthetic data under different gravities (Earth ≈ 9.8, Mars ≈ 3.7, Jupiter ≈ 24.8) and train a neural network to approximate the mapping:

**(Height, Gravity) → Time**

## Installation & Run

Clone this repository and install the required Python packages, and run the script:

```bash
git clone https://github.com/zhihengyii23-ui/Free-Fall-MLP.git
cd Free-Fall-MLP
pip install -r requirements.txt
python MLP-FREE-FALL.py
