# Changelog

### v1.0

#### Progress
- Project structure of NAIC deprecated, JD-fastreID project structure now
- Set up on Ubuntu 20.04 server with VMWare Bitfusion GPU
- Successfully run benchmark on datasets Market1501
- Add *extract-feat* option on the inference part

#### New features
- extract-feat: obtain latent features that can be used for classification with FCN. Usage:
```bash
bitfusion run -n 1 -- python tools/train_net.py --config-file ./configs/Market1501/bagtricks_R50.yml --eval-only --extract-feat MODEL.WEIGHTS logs/market1501/bagtricks_R50/model_best.pth MODEL.DEVICE "cuda:0"
```