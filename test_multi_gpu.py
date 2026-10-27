#!/usr/bin/env python3
"""Test script to verify multi-GPU setup for AlphaFold 3."""

import jax
import numpy as np

def test_multi_gpu():
    """Test multi-GPU availability and setup."""
    print("Testing JAX multi-GPU setup...")
    
    # Get all available devices
    devices = jax.local_devices()
    gpu_devices = jax.local_devices(backend='gpu')
    
    print(f"Total devices: {len(devices)}")
    print(f"GPU devices: {len(gpu_devices)}")
    
    if len(gpu_devices) == 0:
        print("❌ No GPU devices found!")
        return False
    elif len(gpu_devices) == 1:
        print("⚠️  Only 1 GPU found. Multi-GPU not available.")
        print(f"Available GPU: {gpu_devices[0]}")
        return False
    else:
        print(f"✅ Found {len(gpu_devices)} GPUs!")
        for i, device in enumerate(gpu_devices):
            try:
                memory = device.memory_stats().get('bytes_limit', 0) // (1024**3)
                compute_capability = device.compute_capability
                print(f"  GPU {i}: {device} - {memory}GB, compute capability: {compute_capability}")
            except Exception as e:
                print(f"  GPU {i}: {device} - Error getting stats: {e}")
    
    # Test basic multi-device operations
    print("\nTesting multi-device array placement...")
    try:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        
        # Create device mesh
        device_mesh = Mesh(np.array(gpu_devices), axis_names=('gpus',))
        sharding = NamedSharding(device_mesh, P('gpus'))
        
        # Create a simple array and shard it
        x = jax.random.normal(jax.random.PRNGKey(0), (len(gpu_devices) * 1000, 1000))
        x_sharded = jax.device_put(x, sharding)
        
        print(f"✅ Successfully created and sharded array across {len(gpu_devices)} GPUs")
        print(f"Array shape: {x_sharded.shape}, sharding: {x_sharded.sharding}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-device operations: {e}")
        return False

if __name__ == "__main__":
    success = test_multi_gpu()
    if success:
        print("\n✅ Multi-GPU setup appears to be working!")
        print("You can now use --use_multi_gpu=true with run_alphafold.py")
    else:
        print("\n❌ Multi-GPU setup issues detected.")
        print("Single GPU mode will be used by default.")