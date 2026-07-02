# PyTorch Tutorials — Visual Learning Path

**55+ hands-on notebooks** with explanations, graphs, and runnable code on synthetic data (no downloads). Every component is broken down step by step.

## How to use

1. Work through folders **in order** (01 → 12).
2. Run every cell; tweak numbers and re-run to build intuition.
3. Open in **Google Colab**: replace path in this URL:

```
https://colab.research.google.com/github/bansal1600/pytorch-deep-learning/blob/main/tutorials/01_tensors_and_basics/01_scalar_vector_matrix_tensor.ipynb
```

---

## Curriculum

### 01 — Tensors & basics (8 notebooks)
| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 1 | [01_scalar_vector_matrix_tensor](01_tensors_and_basics/01_scalar_vector_matrix_tensor.ipynb) | 0-D → 3-D tensors with diagrams |
| 2 | [02_creating_tensors](01_tensors_and_basics/02_creating_tensors.ipynb) | zeros, rand, arange, linspace, NumPy |
| 3 | [03_tensor_dtypes_and_devices](01_tensors_and_basics/03_tensor_dtypes_and_devices.ipynb) | float32/64, CPU vs GPU |
| 4 | [04_indexing_slicing_broadcasting](01_tensors_and_basics/04_indexing_slicing_broadcasting.ipynb) | Slicing & broadcasting visuals |
| 5 | [05_elementwise_vs_matrix_ops](01_tensors_and_basics/05_elementwise_vs_matrix_ops.ipynb) | `*` vs `@` heatmaps |
| 6 | [06_reshape_view_contiguous](01_tensors_and_basics/06_reshape_view_contiguous.ipynb) | reshape, view, memory layout |
| 7 | [07_tensor_aggregation](01_tensors_and_basics/07_tensor_aggregation.ipynb) | sum/mean along dimensions |
| 8 | [tensors_in_pytorch](01_tensors_and_basics/tensors_in_pytorch.ipynb) | Original quick intro |

### 02 — Autograd (6 notebooks)
| # | Notebook | Graphs |
|---|----------|--------|
| 1 | [01_computation_graph](02_autograd/01_computation_graph.ipynb) | Graph nodes for y=x² |
| 2 | [02_forward_backward_pass](02_autograd/02_forward_backward_pass.ipynb) | Forward values + backward grads |
| 3 | [03_gradients_multivariable](02_autograd/03_gradients_multivariable.ipynb) | Contour + gradient arrows |
| 4 | [04_detach_and_no_grad](02_autograd/04_detach_and_no_grad.ipynb) | When gradients stop |
| 5 | [05_gradient_accumulation](02_autograd/05_gradient_accumulation.ipynb) | Accumulated grad bar chart |
| 6 | [pytorch_autograd](02_autograd/pytorch_autograd.ipynb) | Original autograd intro |

### 03 — nn.Module building blocks (7 notebooks)
| # | Notebook | Graphs |
|---|----------|--------|
| 1 | [01_parameters_and_buffers](03_nn_modules/01_parameters_and_buffers.ipynb) | Parameter counts |
| 2 | [02_linear_layer_math](03_nn_modules/02_linear_layer_math.ipynb) | W, b heatmaps |
| 3 | [03_activation_functions](03_nn_modules/03_activation_functions.ipynb) | ReLU, Sigmoid, Tanh, GELU curves |
| 4 | [04_loss_functions](03_nn_modules/04_loss_functions.ipynb) | MSE, BCE, CE curves |
| 5 | [05_optimizers_compared](03_nn_modules/05_optimizers_compared.ipynb) | SGD vs Adam on contours |
| 6 | [06_building_blocks_sequential](03_nn_modules/06_building_blocks_sequential.ipynb) | Layer shape flow |
| 7 | [pytorch_nn_module](03_nn_modules/pytorch_nn_module.ipynb) | Original nn.Module intro |

### 04 — Training pipeline (6 notebooks)
| # | Notebook | Graphs |
|---|----------|--------|
| 1 | [01_dataset_and_dataloader](04_training/01_dataset_and_dataloader.ipynb) | Batch visualization |
| 2 | [02_train_val_test_split](04_training/02_train_val_test_split.ipynb) | Split pie chart |
| 3 | [03_training_loop_anatomy](04_training/03_training_loop_anatomy.ipynb) | Loss per step |
| 4 | [04_learning_curves](04_training/04_learning_curves.ipynb) | Train vs val curves |
| 5 | [05_batch_size_effect](04_training/05_batch_size_effect.ipynb) | Batch size vs stability |
| 6 | [pytorch_training_pipeline](04_training/pytorch_training_pipeline.ipynb) | Original pipeline |

### 05 — End-to-end projects (3 notebooks)
| Notebook | Project |
|----------|---------|
| [mnist_ann](05_projects/mnist_ann.ipynb) | MNIST with ANN |
| [fashion_mnist_cnn](05_projects/fashion_mnist_cnn.ipynb) | Fashion-MNIST CNN |
| [fashion_mnist_ann_optuna](05_projects/fashion_mnist_ann_optuna.ipynb) | Hyperparameter tuning |

### 06 — CNN basics (5 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_what_is_convolution](06_cnn_basics/01_what_is_convolution.ipynb) | Kernel sliding on signal |
| [02_filters_edge_detection](06_cnn_basics/02_filters_edge_detection.ipynb) | Sobel feature maps |
| [03_pooling_max_vs_avg](06_cnn_basics/03_pooling_max_vs_avg.ipynb) | Max vs avg pool |
| [04_padding_and_stride](06_cnn_basics/04_padding_and_stride.ipynb) | Output size formula |
| [05_mini_cnn_mnist](06_cnn_basics/05_mini_cnn_mnist.ipynb) | Mini CNN + filters |

### 07 — Visualization (5 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_plotting_training_metrics](07_visualization/01_plotting_training_metrics.ipynb) | Loss & accuracy plots |
| [02_weight_histograms](07_visualization/02_weight_histograms.ipynb) | Weight distributions |
| [03_activation_distributions](07_visualization/03_activation_distributions.ipynb) | Per-layer histograms |
| [04_decision_boundary_2d](07_visualization/04_decision_boundary_2d.ipynb) | 2D classifier boundary |
| [05_confusion_matrix](07_visualization/05_confusion_matrix.ipynb) | Confusion heatmap |

### 08 — Optimization landscapes (4 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_gradient_descent_1d](08_optimization_landscapes/01_gradient_descent_1d.ipynb) | GD steps on curve |
| [02_learning_rate_effect](08_optimization_landscapes/02_learning_rate_effect.ipynb) | LR comparison |
| [03_momentum_and_adam](08_optimization_landscapes/03_momentum_and_adam.ipynb) | Optimizer paths on contour |
| [04_local_minima_saddle](08_optimization_landscapes/04_local_minima_saddle.ipynb) | Saddle point |

### 09 — Regularization (4 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_overfitting_polynomial](09_regularization/01_overfitting_polynomial.ipynb) | High-degree overfit |
| [02_dropout_visual](09_regularization/02_dropout_visual.ipynb) | Dropped neurons |
| [03_batch_norm_effect](09_regularization/03_batch_norm_effect.ipynb) | Before/after BN |
| [04_weight_decay](09_regularization/04_weight_decay.ipynb) | L2 weight shrinkage |

### 10 — Sequence models (3 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_sequences_and_rnn_unfold](10_sequence_models/01_sequences_and_rnn_unfold.ipynb) | RNN hidden state over time |
| [02_lstm_gates_intuition](10_sequence_models/02_lstm_gates_intuition.ipynb) | LSTM gate bars |
| [03_attention_weights_heatmap](10_sequence_models/03_attention_weights_heatmap.ipynb) | Attention heatmap |

### 11 — Probability & sampling (2 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_softmax_temperature](11_probability/01_softmax_temperature.ipynb) | Temperature bar charts |
| [02_sampling_strategies](11_probability/02_sampling_strategies.ipynb) | Argmax vs top-k |

### 12 — Data pipeline (2 notebooks)
| Notebook | Graphs |
|----------|--------|
| [01_normalization_standardization](12_data_pipeline/01_normalization_standardization.ipynb) | Histogram before/after |
| [02_train_augmentation_demo](12_data_pipeline/02_train_augmentation_demo.ipynb) | Augmented image grid |

---

## After tutorials

Continue with **[TorchLeet interview prep](../notebooks/torchleet/README.md)** (75 problems with solutions).

## Regenerate tutorials

```bash
python scripts/generate_tutorials_part1.py
python scripts/generate_all_tutorials.py
python scripts/generate_tutorials_extra.py
```

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```
