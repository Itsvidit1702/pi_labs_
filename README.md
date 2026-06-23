# Machine Unlearning Project

This repository focuses on **Machine Unlearning** techniques, exploring methods to efficiently remove the influence of specific training data (such as user-specific data or forgotten cohorts) from trained models without requiring a full retrain from scratch. 

The project tracks model states, computed Fisher Information matrices, and evaluating metrics before and after the unlearning process.

---

## 📁 Repository Structure

```text
machine-unlearning-project/
├── employee_model/              # Base model trained on initial dataset
│   └── checkpoint-375/          # Specific fine-tuning checkpoint
│       ├── model.safetensors
│       └── optimizer.pt
├── fisher_unlearned_model/      # Model unlearned using Fisher Information methods
│   └── checkpoint-10/
│       ├── model.safetensors
│       └── optimizer.pt
├── unlearned_model/             # Alternative baseline unlearned model
│   └── checkpoint-10/
│       ├── model.safetensors
│       └── optimizer.pt
├── fisher_scores.pt             # Cached Fisher Information scores for weights
├── .gitattributes               # Git LFS configuration file
└── .gitignore                  # Git untracked file configurations
