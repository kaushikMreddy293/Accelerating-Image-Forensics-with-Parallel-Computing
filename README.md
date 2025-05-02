# Accelerating Image Forensics with Parallel Computing

---

## Introduction

The rising use of AI-generated images (deepfakes) poses ethical, cybersecurity, and media challenges. Real-time detection is critical. This project explores the use of parallel computing (on CPU & GPU) to speed up deepfake detection using image classification models.

---

## Objective

Utilize parallel computing techniques to distinguish real vs AI-generated images efficiently, reducing computation time while maintaining or improving model accuracy.

---

## Dataset

| Category         | Description                         |
|------------------|-------------------------------------|
| Sources          | Real: Shutterstock, Fake: GAN models |
| Total Size       | ~11.5 GB                            |
| Training Images  | ~80,000                             |
| Testing Images   | ~5,500                              |
| Image Resolution | ~700x700                            |
| Labels           | 0 = Real, 1 = AI-generated          |

---

## Methodology

### Models:
- ResNet18 (custom trained)
- Vision Transformer (ViT)

### Preprocessing:
- ResNet: Resize to 224x224
- ViT: Normalization using ViT-specific mean/std

### Splits:
- ResNet: 50k train / 15k val / 15k test
- ViT: 40k train / 5k val / 5k test

### Evaluations:
- Training time
- Speedup ratios
- GPU/CPU memory usage
- Accuracy
- Parallelism strategies (AMP, DDP, Model Parallelism)

---

## Environment

| Component     | Specs                         |
|---------------|-------------------------------|
| GPU Types     | P100, V100-SXM2               |
| CPUs Tested   | 1, 2, 4, 8                     |
| GPUs Tested   | 1, 2, 4                        |
| Framework     | PyTorch 2.5.1 + cu121         |
| Language      | Python 3.11.11                |

---

## Experiments & Results

### CPU Parallelism for Model Training (ResNet18)
- Accuracy stable
- Speedup peaks at 2 threads
- Synchronization overhead after 2 threads

### CPU Parallelism for Data Loading (ViT)
- 2 workers = best performance
- Diminished speedup beyond 2 due to I/O bottlenecks

### GPU DDP (Distributed Data Parallel)
- ResNet18: Max speedup: 1.11× at 2 GPUs
- ViT: Best speedup: 2.25× on 4 GPUs; accuracy improved with GPU count

### AMP (Automatic Mixed Precision)
- ViT: 28% decrease in training time (353s → 254s)
- ResNet: ~18% increase in training time due to conversion overhead

### Model Parallelism (ResNet18)
- Accuracy increased by 0.17%
- Memory usage decreased by 8%
- Training time unchanged due to GPU communication

---

## Key Observations & Graphs

- CPU Parallelism vs Model Training & Data Loading Speedup (ResNet18, ViT)
  -  <img width="487" alt="image" src="https://github.com/user-attachments/assets/d91413e7-0c9a-4dcd-b2fa-f532b75dae53" />
  - <img width="667" alt="image" src="https://github.com/user-attachments/assets/c6ef10b6-f9ce-4abf-ba43-1aebfef4a5c7" />


- GPU Count vs Speedup in Distributed Data Parallel (ViT & ResNet18)
  - <img width="845" alt="image" src="https://github.com/user-attachments/assets/c49134a1-194f-4e81-9ab7-47fc35b44def" />
  - <img width="800" alt="image" src="https://github.com/user-attachments/assets/f95a09bd-d118-468c-ba39-f09b6a71487f" />
  

- AMP Time Reduction Comparison
  - <img width="989" alt="image" src="https://github.com/user-attachments/assets/0a4c6ace-c6fd-469b-a42d-892e82930108" />
  
- Model Comparison: ViT vs ResNet on GPU scaling
  - <img width="989" alt="image" src="https://github.com/user-attachments/assets/f573f57d-1970-4d68-8a27-a4723e683ee3" />
  


---

## Observations

1. ViT scaled significantly better on GPUs than ResNet18.
2. ResNet18 retained better accuracy on small GPU configurations.
3. AMP and DDP offered the most practical speedups.
4. CPU parallelism offers limited gains beyond 2 threads.

---

## References

- Tidio AI Detection Blog: [Tidio Blog](https://www.tidio.com/blog/ai-test/)
- Human vs AI-Generated Images: [arXiv:2412.09715](https://arxiv.org/abs/2412.09715)
- PyTorch Documentation: [PyTorch.org](https://pytorch.org/docs/stable/index.html)
- Vision Transformers Article: [Vision Transformers - Wolfe](https://cameronrwolfe.substack.com/p/vision-transformers)
- ResNet Deepfake Detection: [Springer](https://link.springer.com/article/10.1007/s00371-024-03613-x)

---

## Conclusion

- CPU parallelism is helpful up to 2 threads but has diminishing returns.
- GPU DDP training shows significant performance boost with ViT.
- AMP reduces memory and improves speed with minimal accuracy loss.
- Model Parallelism is memory-efficient but not time-efficient.
