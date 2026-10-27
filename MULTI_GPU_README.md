# Multi-GPU Support for AlphaFold 3

## Overview

The AlphaFold 3 script has been modified to support multi-GPU inference for increased memory capacity. This allows running larger models or larger inputs by distributing the computational load and memory usage across multiple GPUs.

## New Command Line Options

### `--use_multi_gpu`
- **Type**: Boolean flag
- **Default**: `False`
- **Description**: Enable multi-GPU mode for model parallelism to increase available memory
- **Usage**: `--use_multi_gpu=true`

### `--num_gpus`
- **Type**: Integer
- **Default**: `None` (uses all available GPUs)
- **Description**: Specify the number of GPUs to use for multi-GPU inference
- **Usage**: `--num_gpus=4`

## How It Works

1. **Parameter Sharding**: Model parameters are distributed (sharded) across multiple GPUs along the first dimension, reducing per-GPU memory requirements for the model weights.

2. **Data Replication**: Input data is replicated across all GPUs to ensure each GPU has access to the complete input.

3. **Computation Distribution**: The forward pass computation is distributed across GPUs, with JAX handling the coordination.

4. **Memory Pooling**: Effectively pools the memory of multiple GPUs, allowing for larger models or inputs than would fit on a single GPU.

## Usage Examples

### Basic Multi-GPU Usage
```bash
# Use all available GPUs
python run_alphafold.py \
    --json_path=input.json \
    --output_dir=output \
    --use_multi_gpu=true

# Use specific number of GPUs
python run_alphafold.py \
    --json_path=input.json \
    --output_dir=output \
    --use_multi_gpu=true \
    --num_gpus=4
```

### Single GPU (Original Behavior)
```bash
# Default single GPU mode
python run_alphafold.py \
    --json_path=input.json \
    --output_dir=output

# Specify which GPU to use in single GPU mode
python run_alphafold.py \
    --json_path=input.json \
    --output_dir=output \
    --gpu_device=1
```

## System Requirements

### Hardware
- Multiple NVIDIA GPUs with compute capability ≥ 6.0
- Sufficient GPU memory (8GB+ per GPU recommended)
- PCIe connectivity between GPUs (NVLink preferred for best performance)

### Software
- CUDA-capable JAX installation
- NVIDIA drivers supporting your GPU generation
- Sufficient system memory for data loading

## Testing Multi-GPU Setup

Use the included test script to verify your multi-GPU setup:

```bash
python test_multi_gpu.py
```

This will:
- Check available GPU devices
- Test basic multi-device operations
- Verify JAX can create device meshes and shard arrays

## Performance Considerations

### When to Use Multi-GPU
- **Large protein complexes** with many chains/residues
- **Memory constraints** on single GPU runs
- **Large MSA sizes** (>1024 sequences)
- **Multiple diffusion samples** with high recycle counts

### When Single GPU May Be Better
- **Small to medium proteins** (<500 residues)
- **Fast prototyping** and testing
- **Limited inter-GPU bandwidth** (e.g., PCIe only)

### Expected Benefits
- **Memory capacity**: Approximately linear scaling with number of GPUs
- **Throughput**: May vary depending on model size and inter-GPU communication
- **Stability**: Larger memory headroom reduces OOM errors

## Troubleshooting

### Common Issues

1. **"No GPU devices found"**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import jax; print(jax.devices())"
   ```

2. **"GPU compute capability too low"**
   - Ensure all GPUs have compute capability ≥ 6.0
   - Use `nvidia-smi` to check your GPU models

3. **"Out of memory" even with multi-GPU**
   - Try reducing `--num_msa` parameter
   - Reduce `--num_diffusion_samples`
   - Check if all GPUs are being used: watch `nvidia-smi`

4. **Slower than single GPU**
   - Check inter-GPU bandwidth
   - Consider using fewer GPUs if communication overhead is high
   - Verify GPU topology: `nvidia-smi topo -m`

### Environment Variables

For better memory management:
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For GPU compute capability 7.x:
```bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
```

## Monitoring

### GPU Usage
```bash
# Monitor GPU utilization and memory
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -q -d UTILIZATION,MEMORY,TEMPERATURE
```

### JAX Device Status
```python
import jax
print("Available devices:", jax.devices())
print("Local devices:", jax.local_devices())
```

## Advanced Configuration

### Custom Sharding Strategies
The current implementation uses simple model parallelism. For more advanced use cases, you may want to modify the sharding strategy in the `ModelRunner.__init__()` method.

### Memory Optimization
- Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` based on your system
- Consider using gradient checkpointing for even lower memory usage
- Monitor actual memory usage patterns with `nvidia-smi`

## Compatibility

- **AlphaFold 3**: All model versions
- **JAX**: Requires JAX with multi-GPU support
- **GPU Generation**: NVIDIA GPUs with compute capability ≥ 6.0
- **CUDA**: Compatible CUDA version for your hardware

## Support

If you encounter issues:
1. Run the test script: `python test_multi_gpu.py`
2. Check GPU compatibility and memory
3. Verify JAX installation supports multi-GPU
4. Review environment variable settings