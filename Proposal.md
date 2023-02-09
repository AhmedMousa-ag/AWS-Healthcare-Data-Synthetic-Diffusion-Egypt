# Data Synthetics Project

## Domain Background

**Applicable Domains:** Data Synthetics can be applied to almost all domains. But in this project will be applied to **Healthcare** domain.

**Historical Information:** The idea of original fully synthetic data was created by **Donald Bruce Rubin**. Rubin originally designed this to synthesize the Decennial Census long form responses for the short form households to reserve their privacy. and it can be applied to **Healthcare** domain.

## Problem Statement

**What's the problem?.** When you visit the hospital for a health issue or regular check, you always give the hospital important information about your self *e.g.,* your diet, medicines you use, health concerns, sensitive data about yourself .etc.

But you want to preserve you privacy and don't want your information out? in the same time some scientists needs these information for to help future patience to come up with a cure or something to help others! and of course scientists want to preserve your privacy.

## Solution Statement

**Why Synthetic Data in Healthcare?.** Synthetic data can protect patient privacy and augmenting clinical research as synthetic data is a derivative of the original real data but no synthetic datapoint can be attributed to a single real datapoint.

## Datasets and Inputs

Dataset is a tabular data which contains 60 rows of blood analysis of Rheumatoid arthritis patience. Data set obtained from a hospital in Cairo, Egypt.

Data approved only for the context of this project, and all ids can't be traced to this dataset.

## Benchmark Model

The best known model in Tabular Data Generation  is: [Causal-TGAN](https://arxiv.org/pdf/2104.10680v1.pdf) with a KS test average score of 0.81 for adult, census and news datasets.

## Evaluation Metrics

KS test metrics will be used for evaluation, which is: **Kolmogorov–Smirnov test** which compares the distribution of one or two samples.

## Project Design
