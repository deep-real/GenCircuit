CUDA_VISIBLE_DEVICES=0 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 100 --num-examples 1000000000 --fragment 0
CUDA_VISIBLE_DEVICES=1 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 100 --num-examples 1000000000 --fragment 1
CUDA_VISIBLE_DEVICES=2 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 100 --num-examples 1000000000 --fragment 2
CUDA_VISIBLE_DEVICES=3 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 100 --num-examples 1000000000 --fragment 3
CUDA_VISIBLE_DEVICES=4 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 50 --num-examples 1000000000 --fragment 4
CUDA_VISIBLE_DEVICES=5 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 50 --num-examples 1000000000 --fragment 5
CUDA_VISIBLE_DEVICES=6 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 50 --num-examples 1000000000 --fragment 6
CUDA_VISIBLE_DEVICES=7 python run_attribution.py --models ViT-B_16 --tasks waterbirds-bg-water-group --ablation patching --method exact-optimized-parallel --batch-size 50 --num-examples 1000000000 --fragment 7

