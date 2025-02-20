# VIP_project

## How to use

- install the dependencies

```pip install -r requirements.txt```

- run the code

```python main```

## History

- 2025/02/17  
  - Create origin version.
  
- 2025/02/17 
  - Modify some dependency formats(torch_xx==xx.xx-cudaxx -> torch_xx==xx.xx). 
  > PyPI no longer supports specifying the CUDA version of Torch.
  - Modify the absolute path in the `main` function to a relative path.(Line:261)
  - Modify the `harzard_preds` in the `train_model` and `test_model` functions. (Line:170,204)
  > TypeError: ResNet.forward() takes 2 positional arguments but 4 were given.
  - Modify the storage directory for epochs to `\PROJECT\epochs\epoch_num.pt`.(Line:157)

- 2025/02/19
  - Modify loss caculation function to cross entropy in `train_model` function.(Line:177)
  > Loss of Resnet is nan.