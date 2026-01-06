import torch
import torchvision

print("="*80)
print("GPU SETUP VERIFICATION")
print("="*80)

# Check PyTorch version
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ Torchvision version: {torchvision.__version__}")

# Check CUDA availability
print(f"\n✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Test GPU operation
    print(f"\n✓ Current device: {torch.cuda.current_device()}")
    
    # Create a tensor on GPU
    x = torch.rand(3, 512, 683).cuda()
    print(f"✓ Successfully created tensor on GPU: {x.device}")
    
    print("\n✅ GPU is ready for training!")
else:
    print("\n⚠️  CUDA not available. Training will use CPU (slower).")
    print("   To enable GPU:")
    print("   1. Install NVIDIA drivers")
    print("   2. Install CUDA toolkit")
    print("   3. Reinstall PyTorch with CUDA support")

print("="*80)
