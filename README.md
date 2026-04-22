
# Cross-Attention Conditional Diffusion for Magnetic Flux Leakage Signal Synthesis
## 📄 Overview
Magnetic flux leakage (MFL) inspection is a widely used nondestructive technique for detecting pipeline corrosion and metal loss, but its effectiveness is limited by the scarcity of diverse, high-quality data, since both field measurements and physics-based simulations are costly. This study proposes a cross-attention conditional diffusion model (XCDiff), guided by pipeline defect geometry parameters, to synthesize high-fidelity MFL signals. XCDiff employs a geometry-aware U-Net denoiser in which cross-attention fuses defect geometry embeddings into feature maps at every encoder-decoder stage, enabling the denoising process to adapt to varying defect configurations and produce geometry-consistent multi-channel magnetic field responses. The trained XCDiff model is used to generate augmented MFL samples for training a downstream ResNet to predict defect depth, width, and length. As more XCDiff-augmented data are incorporated, the downstream regression model achieves higher accuracy and reduced predictive uncertainty, reflected by progressively narrower prediction intervals. These results demonstrate that diffusion-based augmentation with XCDiff improves both predictive performance and uncertainty quantification in MFL-based pipeline defect assessment.

## 🧠 Method
- Conditional diffusion model for MFL signal generation  
- U-Net backbone with **cross-attention (bottleneck & feature fusion)**  
- Geometry-conditioned generation (e.g., depth, width, length)  
- Time-step embedding for diffusion process  

## 🖼️ Model Architecture

<p align="center">
  <img width="865" height="696" alt="image" src="https://github.com/user-attachments/assets/dd14f59e-8010-43d6-83a3-c8f01a5cd301" />
</p>

---

## 📂 Project Structure

```bash
├── csv/                     # Input data (CSV format)
├── gen/                     # Generated results
├── images/                  # Visualization outputs
├── logs/                    # Training logs & loss curves
├── weight/                  # Model checkpoints

├── train.py                 # Training entry
├── inf.py                   # Batch inference (CSV input)
├── inference_once.py        # Single inference (manual input)
├── diffusion.py             # Diffusion visualization

├── dataset.py               # Dataset loader
├── config.py                # Configuration
├── denoise.py               # Denoising process
├── conv_block.py            # Convolution blocks
├── time_position_emb.py     # Time embedding
├── cross_attn.py            # Cross-attention module

├── stable_unet.py
├── stable_unet_noattn.py
├── stable_unet_bottleneck_ca.py  # ⭐ main model

├── image_metrics.py         # Evaluation metrics
├── average_metrics.json     # Experiment results



If you use this code, please cite:
@article{wu2026xcdiff,
  title={Cross-attention conditional diffusion for magnetic flux leakage signal synthesis and pipeline defect uncertainty analysis},
  author={Wu, Hao and others},
  journal={Measurement},
  year={2026},
  pages={121208}
}Wu, Hao, et al. "Cross-attention conditional diffusion for magnetic flux leakage signal synthesis and pipeline defect uncertainty analysis." Measurement (2026): 121208.

