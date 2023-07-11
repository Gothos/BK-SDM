import torch
import diffusers
import transformers
class BKSDM():
    def __init__(self,model_id="runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16,model_type='base',device='cuda'):
        self.pipe=diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe.to(device)
        if model_type=='base':
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
        elif model_type=='small' or model_type=='midless':
            self.pipe.unet.mid_block=None
            if model_type=='small':
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
        
    def __call__(self,prompt,num_inference_steps= 50, guidance_scale=7.5, negative_prompt= None):
      return self.pipe(prompt,num_inference_steps= num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt)  
