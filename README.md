# BK-SDM
An unofficial implementation of [BK-SDM](https://arxiv.org/abs/2305.15798).<br>
The modes "base","small" and "tiny" all require distillation to produce coherent outputs, mode 'midless' is usable out of the box.
The Stablediff_to_BKSDM class takes in a model card and the type of model to reduce to, and loads and converts a pipeline as its *pipe* attribute<br>
```python
model_type='tiny' # One of 'midless'/'base'/small'/'tiny'
model_card=''  # Model Card of the U-net from huggingface
bksdm= Stablediff_to_BKSDM(model_card)
pipe=BKSDM.pipe
# Now call pipe as you would run any other Stable Diffusion pipeline
```
You can also convert a unet to one of the BK-SDM U-nets by calling the unetprep() function on it, which prepares a unet as per the model type given, in place.
```python
model_type='tiny' # One of 'midless'/'base'/small'/'tiny'
Unet= # Load Pytorch U-net from SD model
unetprep(Unet,model_type)
# Now Unet has been converted to a BK-SDM style tiny U-net
```
