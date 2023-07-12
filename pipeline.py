import torch
import gc
import diffusers
import transformers
class Stablediff_to_BKSDM():
    def __init__(self,model_id="runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16,model_type='base',device='cuda',**kwargs):
        if not torch.cuda.is_available():
            device='cpu'
            torch_dtype=torch.float32
        self.pipe=diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.pipe.to(device)
        if model_type!='base':
          self.pipe.unet.mid_block=None
        if model_type!='midless':
          for i in range(3):
              delattr(self.pipe.unet.down_blocks[i].resnets,'1')
              delattr(self.pipe.unet.down_blocks[i].attentions,'1')
          if model_type=='tiny':
              delattr(self.pipe.unet.down_blocks,'3')
              self.pipe.unet.down_blocks[2].downsamplers=None
          else:
              delattr(self.pipe.unet.down_blocks[3].resnets,'1')
          if model_type=='tiny':
              self.pipe.unet.up_blocks[0]=self.pipe.unet.up_blocks[1]
              self.pipe.unet.up_blocks[1]=self.pipe.unet.up_blocks[2]
              self.pipe.unet.up_blocks[2]=self.pipe.unet.up_blocks[3]
              delattr(self.pipe.unet.up_blocks,'3')   
      else:
          self.pipe.unet.up_blocks[0].resnets[1]=self.pipe.unet.up_blocks[0].resnets[2]
          delattr(self.pipe.unet.up_blocks[0].resnets,'2')
          
          delattr(self.pipe.unet.down_blocks[3].resnets,'1')
          self.pipe.unet.up_blocks[0].resnets[1]=self.pipe.unet.up_blocks[0].resnets[2]
          delattr(self.pipe.unet.up_blocks[0].resnets,'2')
          self.pipe.unet.up_blocks[1].attentions[1]=self.pipe.unet.up_blocks[1].attentions[2]
          delattr(self.pipe.unet.up_blocks[1].attentions,'2')
          delattr(self.pipe.unet.up_blocks[1].resnets,'1')
          self.pipe.unet.up_blocks[2].attentions[1]=self.pipe.unet.up_blocks[2].attentions[2]
          self.pipe.unet.up_blocks[2].resnets[1]=self.pipe.unet.up_blocks[2].resnets[2]
          delattr(self.pipe.unet.up_blocks[2].attentions,'2')
          delattr(self.pipe.unet.up_blocks[2].resnets,'2')
          self.pipe.unet.up_blocks[3].attentions[1]=self.pipe.unet.up_blocks[3].attentions[2]
          self.pipe.unet.up_blocks[3].resnets[1]=self.pipe.unet.up_blocks[3].resnets[2]
          delattr(self.pipe.unet.up_blocks[3].attentions,'2')
          delattr(self.pipe.unet.up_blocks[3].resnets,'2')
        torch.cuda.empty_cache()
        gc.collect()

    def __call__(self,prompt,num_inference_steps= 50, guidance_scale=7.5, negative_prompt= None):
      return self.pipe(prompt,num_inference_steps= num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt)
def unetprep(unet,model_type):
    if model_type!='base':
          unet.mid_block=None
    if model_type!='midless':
      for i in range(3):
          delattr(unet.down_blocks[i].resnets,'1')
          delattr(unet.down_blocks[i].attentions,'1')
      if model_type=='tiny':
          delattr(unet.down_blocks,'3')
          unet.down_blocks[2].downsamplers=None
      else:
          delattr(unet.down_blocks[3].resnets,'1')
      if model_type=='tiny':
          unet.up_blocks[0]=unet.up_blocks[1]
          unet.up_blocks[1]=unet.up_blocks[2]
          unet.up_blocks[2]=unet.up_blocks[3]
          delattr(unet.up_blocks,'3')   
      else:
          unet.up_blocks[0].resnets[1]=unet.up_blocks[0].resnets[2]
          delattr(unet.up_blocks[0].resnets,'2')
      unet.up_blocks[1].attentions[1]=unet.up_blocks[1].attentions[2]
      delattr(unet.up_blocks[1].attentions,'2')
      delattr(unet.up_blocks[1].resnets,'1')
      unet.up_blocks[2].attentions[1]=unet.up_blocks[2].attentions[2]
      unet.up_blocks[2].resnets[1]=unet.up_blocks[2].resnets[2]
      delattr(unet.up_blocks[2].attentions,'2')
      delattr(unet.up_blocks[2].resnets,'2')
      unet.up_blocks[3].attentions[1]=unet.up_blocks[3].attentions[2]
      unet.up_blocks[3].resnets[1]=unet.up_blocks[3].resnets[2]
      delattr(unet.up_blocks[3].attentions,'2')
      delattr(unet.up_blocks[3].resnets,'2')
    torch.cuda.empty_cache()
    gc.collect()
    
